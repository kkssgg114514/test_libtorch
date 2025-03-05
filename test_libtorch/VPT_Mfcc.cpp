#include "VPT_Mfcc.h"
#include "wavreader.h"
#include <librosa/librosa.h>

#include <iostream>
#include <vector>

#include <chrono>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include "Filesearch.h"

void VPT_Mfcc::extractMfccFeatures(std::string url, int frame_length, int frame_space)
{
	//读取样本文件
	void* h_x = wav_read_open(url.c_str());
	//提取样本文件的数据(文件头)
	int res = wav_get_header(h_x, &format, &channels, &sr, &bits_per_sample, &data_length);
	if (!res)
	{
		std::cerr << "get ref header error: " << res << std::endl;
		return;
	}
	//计算样本个数
	samples = data_length * 8 / bits_per_sample;
	//将样本从整型数组存储转到浮点型数组(原本是void)
	std::vector<int16_t> tmp(samples);
	res = wav_read_data(h_x, reinterpret_cast<unsigned char*>(tmp.data()), data_length);
	if (res < 0)
	{
		std::cerr << "read wav file error: " << res << std::endl;
		return;
	}
	std::vector<float> x(samples);
	std::transform(tmp.begin(), tmp.end(), x.begin(),
				   [] (int16_t a)
				   {
					   return static_cast<float>(a) / 32767.f;
				   });

	n_fft = sr / 1000 * frame_length;
	n_hop = sr / 1000 * frame_space;
	fmax = sr / 2;

	mfcc_features = librosa::Feature::mfcc(
		x,
		sr,
		n_fft,
		n_hop,
		"hann",
		true,
		"reflect",
		2.0,
		n_mel,
		fmin,
		fmax,
		80,
		true,
		2
	);
}

void VPT_Mfcc::saveMfccFeatures(const std::string& output_path)
{
	// 使用二进制写入模式保存特征
	std::ofstream out_file(output_path, std::ios::binary);
	if (!out_file) {
		std::cerr << "无法创建特征文件: " << output_path << std::endl;
		return;
	}

	// 保存特征维度信息
	size_t rows = mfcc_features.size();  // 特征向量的数量
	out_file.write(reinterpret_cast<char*>(&rows), sizeof(size_t));

	// 如果特征为空，直接返回
	if (rows == 0) {
		out_file.close();
		return;
	}

	// 保存每个特征向量的长度
	size_t cols = mfcc_features[0].size();
	out_file.write(reinterpret_cast<char*>(&cols), sizeof(size_t));

	// 逐行写入特征数据
	for (const auto& feature_vector : mfcc_features) {
		out_file.write(reinterpret_cast<const char*>(feature_vector.data()),
			feature_vector.size() * sizeof(float));
	}

	out_file.close();
}

void VPT_Mfcc::processBatchMfccFeatures(const std::string& input_dir, const std::string& output_dir)
{
	// 确保输出目录存在
	std::filesystem::create_directories(output_dir);

	// 查找所有WAV文件
	auto wav_files = Filesearch::getAllFiles(input_dir);

	for (const auto& wav_file : wav_files) {
		try {
			// 重置特征矩阵
			mfcc_features.clear();

			// 提取特征
			extractMfccFeatures(wav_file, 25, 10);

			// 构建相对路径
			std::filesystem::path input_path(wav_file);
			std::filesystem::path relative_path =
				std::filesystem::relative(input_path, input_dir);

			// 构建对应的输出路径
			std::filesystem::path output_path =
				std::filesystem::path(output_dir) /
				relative_path.parent_path() /
				(input_path.stem().string() + ".feature");

			// 确保输出目录存在
			std::filesystem::create_directories(output_path.parent_path());

			// 保存特征
			saveMfccFeatures(output_path.string());

			std::cout << "处理成功: " << input_path.filename()
				<< " -> " << output_path.filename() << std::endl;
		}
		catch (const std::exception& e) {
			std::cerr << "处理文件时发生错误: " << wav_file
				<< ", 错误信息: " << e.what() << std::endl;
		}
	}
}

void VPT_Mfcc::loadMfccFeatures(const std::string& input_path)
{
	std::ifstream in_file(input_path, std::ios::binary);
	if (!in_file) {
		std::cerr << "无法打开特征文件: " << input_path << std::endl;
		return;
	}

	// 读取特征维度信息
	size_t rows, cols;
	in_file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
	in_file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));

	// 清空并调整mfcc_features大小
	mfcc_features.clear();
	mfcc_features.resize(rows, std::vector<float>(cols));

	// 逐行读取特征数据
	for (auto& feature_vector : mfcc_features) {
		in_file.read(reinterpret_cast<char*>(feature_vector.data()),
			feature_vector.size() * sizeof(float));
	}

	in_file.close();
}
