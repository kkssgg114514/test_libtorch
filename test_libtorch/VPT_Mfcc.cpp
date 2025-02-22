#include "VPT_Mfcc.h"
#include "wavreader.h"
#include <librosa/librosa.h>

#include <iostream>
#include <vector>

#include <chrono>
#include <numeric>
#include <algorithm>

void VPT_Mfcc::extractMfccFeatures(std::string url)
{
	//读取样本文件
	void* h_x = wav_read_open("../samples/p225_002.wav");
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
