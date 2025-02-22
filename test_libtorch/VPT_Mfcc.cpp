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
	//��ȡ�����ļ�
	void* h_x = wav_read_open("../samples/p225_002.wav");
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
