#pragma once
#include <vector>
#include <string>

class VPT_Mfcc
{
public:
	//ʹ��ʱ���������ļ���
	void extractMfccFeatures(std::string url);


private:

	//�ļ�ͷ��������Ϣ
	int format; 
	//��������
	int channels;
	//Ƶ��
	int sr;
	//ÿ��������λ��
	int bits_per_sample;

	//���ݳ���
	unsigned int data_length;

	//��������
	int samples;

	//����������Ҫ�Ĳ���

	//fft����,Խ��Ƶ�ʷֱ���Խ��
	int n_fft = 400;
	//֡��,ͨ��Ϊfft��1/4
	int n_hop = 100;
	//������������,����ͨ��֡���ں�֡���,������������,����������msΪ��λ

	//÷���˲����������
	int n_mel = 128;

	//Ƶ�ʷ�Χ,÷���˲�������Ƶ��������,ͨ����0~�ο�˹��Ƶ��
	int fmin = 0;
	int fmax = 7600;

	//�����ȡ��mfcc����
	std::vector<std::vector<float>> mfcc_features;
};

