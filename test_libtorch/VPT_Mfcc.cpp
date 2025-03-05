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
	//��ȡ�����ļ�
	void* h_x = wav_read_open(url.c_str());
	//��ȡ�����ļ�������(�ļ�ͷ)
	int res = wav_get_header(h_x, &format, &channels, &sr, &bits_per_sample, &data_length);
	if (!res)
	{
		std::cerr << "get ref header error: " << res << std::endl;
		return;
	}
	//������������
	samples = data_length * 8 / bits_per_sample;
	//����������������洢ת������������(ԭ����void)
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
	// ʹ�ö�����д��ģʽ��������
	std::ofstream out_file(output_path, std::ios::binary);
	if (!out_file) {
		std::cerr << "�޷����������ļ�: " << output_path << std::endl;
		return;
	}

	// ��������ά����Ϣ
	size_t rows = mfcc_features.size();  // ��������������
	out_file.write(reinterpret_cast<char*>(&rows), sizeof(size_t));

	// �������Ϊ�գ�ֱ�ӷ���
	if (rows == 0) {
		out_file.close();
		return;
	}

	// ����ÿ�����������ĳ���
	size_t cols = mfcc_features[0].size();
	out_file.write(reinterpret_cast<char*>(&cols), sizeof(size_t));

	// ����д����������
	for (const auto& feature_vector : mfcc_features) {
		out_file.write(reinterpret_cast<const char*>(feature_vector.data()),
			feature_vector.size() * sizeof(float));
	}

	out_file.close();
}

void VPT_Mfcc::processBatchMfccFeatures(const std::string& input_dir, const std::string& output_dir)
{
	// ȷ�����Ŀ¼����
	std::filesystem::create_directories(output_dir);

	// ��������WAV�ļ�
	auto wav_files = Filesearch::getAllFiles(input_dir);

	for (const auto& wav_file : wav_files) {
		try {
			// ������������
			mfcc_features.clear();

			// ��ȡ����
			extractMfccFeatures(wav_file, 25, 10);

			// �������·��
			std::filesystem::path input_path(wav_file);
			std::filesystem::path relative_path =
				std::filesystem::relative(input_path, input_dir);

			// ������Ӧ�����·��
			std::filesystem::path output_path =
				std::filesystem::path(output_dir) /
				relative_path.parent_path() /
				(input_path.stem().string() + ".feature");

			// ȷ�����Ŀ¼����
			std::filesystem::create_directories(output_path.parent_path());

			// ��������
			saveMfccFeatures(output_path.string());

			std::cout << "����ɹ�: " << input_path.filename()
				<< " -> " << output_path.filename() << std::endl;
		}
		catch (const std::exception& e) {
			std::cerr << "�����ļ�ʱ��������: " << wav_file
				<< ", ������Ϣ: " << e.what() << std::endl;
		}
	}
}

void VPT_Mfcc::loadMfccFeatures(const std::string& input_path)
{
	std::ifstream in_file(input_path, std::ios::binary);
	if (!in_file) {
		std::cerr << "�޷��������ļ�: " << input_path << std::endl;
		return;
	}

	// ��ȡ����ά����Ϣ
	size_t rows, cols;
	in_file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
	in_file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));

	// ��ղ�����mfcc_features��С
	mfcc_features.clear();
	mfcc_features.resize(rows, std::vector<float>(cols));

	// ���ж�ȡ��������
	for (auto& feature_vector : mfcc_features) {
		in_file.read(reinterpret_cast<char*>(feature_vector.data()),
			feature_vector.size() * sizeof(float));
	}

	in_file.close();
}
