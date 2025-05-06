#pragma once
#include <vector>
#include <string>

class VPT_Mfcc
{
public:
	//使用时仅需输入文件名
	void extractMfccFeatures(
		std::string url,	//文件路径
		int frame_length,	//帧长度
		int frame_space		//帧间隔
	);

//保存文件
	void saveMfccFeatures(const std::string& output_path);

	//批量处理文件函数
	void processBatchMfccFeatures(const std::string& input_dir, const std::string& output_dir);

	//读取对应的特征文件函数
	void loadMfccFeatures(const std::string& input_path);

	std::vector<std::vector<float>> getFeature() const;

private:

	//文件头包含的信息
	int format;
	//声道数量
	int channels;
	//频率(采样率)
	int sr;
	//每个采样的位数
	int bits_per_sample;

	//数据长度
	unsigned int data_length;

	//总样本数
	int samples;

	//处理数据需要的参数

	//fft窗口,越大频率分辨率越大
	int n_fft = 400;
	//帧移,通常为fft的1/4
	int n_hop = 100;
	//以上两个数据,可以通过帧窗口和帧间隔,采样率来计算,这两个都是ms为单位

	//梅尔滤波器组的数量
	int n_mel = 128;

	//频率范围,梅尔滤波器组有频率上下限,通常是0~奈奎斯特频率
	int fmin = 0;
	int fmax = 7600;

	//存放提取的mfcc特征
	std::vector<std::vector<float>> mfcc_features;
};
