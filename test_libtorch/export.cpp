#include "export.h"
#include "Filesearch.h"
#include "SpeakerDataSet.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>

void trainModel(std::string wavDir, int index)
{
	//ѵ��˵����ģ��
	SpeakerDataSet speaker_train;
	//ָ���м������ļ����·��
	std::string mid_dir = "..\\featureDir\\" + std::to_string(index);
	speaker_train.processBatchMfccFeatures(wavDir, mid_dir);
	// ������Ŀ¼���浽�ļ���
	std::string feature_path_file = "..\\feaDir.txt";
	Filesearch::generate_feature_paths(mid_dir, feature_path_file, index);
	//speaker_train.load_feature_paths("..\\feaDir.txt");
	//ָ��model·��
	std::string model_dir = "..\\modelDir";
	//ѵ��ģ��
	speaker_train.train_speaker_models(model_dir);

}

void trainModel2(std::string wavDir)
{
	//ѵ��˵����ģ��
	SpeakerDataSet speaker_train;
	//����ƵĿ¼���Ƴ�Ŀ¼�ļ�
	speaker_train.save_audio_to_directory(wavDir);
	//��Ŀ¼�ļ���ȡ·���Ͷ�Ӧid
	speaker_train.load_wav_index();
	//���ö�ȡ��·����id���mfcc����
	speaker_train.processBatchWavMfccFeatures();
	// ѵ��ģ��
	//ָ��model·��
	std::string model_dir = "..\\modelDir";
	speaker_train.train_speaker_models_3(model_dir);
}

int testModel(std::string wavPath)
{
	SpeakerDataSet speaker_test;
	//ָ���м������ļ����·��
	std::string mid_dir = "..\\featureDir";
	// ������Ƶ����mfcc�����ȡ����
	std::string test_feature = speaker_test.processMfccFeatures(wavPath, mid_dir);
	//ȫ��ģ�ʹ洢Ŀ¼
	std::string model_dir = "..\\modelDir";
	//����ģ��
	int user = speaker_test.test_speaker_models_2(model_dir, test_feature);

	return user;
}
