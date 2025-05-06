#include "export.h"
#include "Filesearch.h"
#include "SpeakerDataSet.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>

void trainModel(std::string wavDir, int index)
{
	//训练说话人模型
	SpeakerDataSet speaker_train;
	//指定中间特征文件输出路径
	std::string mid_dir = "..\\featureDir\\" + std::to_string(index);
	speaker_train.processBatchMfccFeatures(wavDir, mid_dir);
	// 将特征目录保存到文件中
	std::string feature_path_file = "..\\feaDir.txt";
	Filesearch::generate_feature_paths(mid_dir, feature_path_file, index);
	//speaker_train.load_feature_paths("..\\feaDir.txt");
	//指定model路径
	std::string model_dir = "..\\modelDir";
	//训练模型
	speaker_train.train_speaker_models(model_dir);

}

void trainModel2(std::string wavDir)
{
	//训练说话人模型
	SpeakerDataSet speaker_train;
	//将音频目录编制成目录文件
	speaker_train.save_audio_to_directory(wavDir);
	//从目录文件读取路径和对应id
	speaker_train.load_wav_index();
	//利用读取的路径和id输出mfcc特征
	speaker_train.processBatchWavMfccFeatures();
	// 训练模型
	//指定model路径
	std::string model_dir = "..\\modelDir";
	speaker_train.train_speaker_models_3(model_dir);
}

int testModel(std::string wavPath)
{
	SpeakerDataSet speaker_test;
	//指定中间特征文件输出路径
	std::string mid_dir = "..\\featureDir";
	// 测试音频经过mfcc处理获取特征
	std::string test_feature = speaker_test.processMfccFeatures(wavPath, mid_dir);
	//全部模型存储目录
	std::string model_dir = "..\\modelDir";
	//测试模型
	int user = speaker_test.test_speaker_models_2(model_dir, test_feature);

	return user;
}
