#include "SpeakerDataSet.h"
#include <iostream>
#include <fstream>

void SpeakerDataSet::processBatchMfccFeatures(const std::string& input_dir, const std::string& output_dir)
{
	mfcc_processer.processBatchMfccFeatures(input_dir, output_dir);
}

SpeakerSample SpeakerDataSet::loadFeatureFile(const std::string& input_path, int64_t id)
{
	SpeakerSample res;
	//样本id设置为指定id
	res.speaker_id = id;
	//提取feature
	mfcc_processer.loadMfccFeatures(input_path);

	//转换Tensor
	res.features = vectorToTensor(mfcc_processer.getFeature());

	return res;
}

void SpeakerDataSet::load_feature_paths(const std::string& path_file)
{
	std::ifstream infile(path_file);
	std::string feature_path;
	int speaker_id;

	while (infile >> feature_path >> speaker_id) {
		feature_paths.push_back(feature_path);
		speaker_ids.push_back(speaker_id);
	}
}

torch::Tensor SpeakerDataSet::vectorToTensor(std::vector<std::vector<float>> feature_sample)
{
	size_t rows = feature_sample.size();
	size_t cols = feature_sample.at(0).size();

	// 创建一个Tensor内存区域
	torch::Tensor tensor = torch::zeros({ static_cast<long>(rows), static_cast<long>(cols) },
		torch::TensorOptions().dtype(torch::kFloat32));

	// 复制数据
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			tensor[i][j] = feature_sample.at(i).at(j);
		}
	}

	return tensor;
}
