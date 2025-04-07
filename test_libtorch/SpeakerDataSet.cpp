#include "SpeakerDataSet.h"
#include "ECAPA_TDNN.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "SpeakerVerificationLoss.h"
#include <random>

void SpeakerDataSet::processBatchMfccFeatures(const std::string& input_dir, const std::string& output_dir)
{
	mfcc_processer.processBatchMfccFeatures(input_dir, output_dir);
}

SpeakerSample SpeakerDataSet::loadFeatureFile(const std::string& input_path, int64_t id)
{
	SpeakerSample res;
	//����id����Ϊָ��id
	res.speaker_id = id;
	//��ȡfeature
	mfcc_processer.loadMfccFeatures(input_path);

	//ת��Tensor
	res.features = vectorToTensor(mfcc_processer.getFeature());

	return res;
}

void SpeakerDataSet::load_feature_paths(const std::string& path_file)
{
	std::ifstream infile(path_file);
	std::string feature_path;
	int speaker_id;

	while (infile >> feature_path >> speaker_id)
	{
		feature_paths.push_back(feature_path);
		speaker_ids.push_back(speaker_id);
	}
}

void SpeakerDataSet::train_speaker_models(const std::string& output_dir)
{
	// ��˵����ID��������
	std::unordered_map<int, std::vector<torch::Tensor>> speaker_features;

	int preid = 0;
	int k = 0;

	for (size_t i = 0; i < feature_paths.size(); ++i)
	{
		if (preid != speaker_ids[i])
		{
			preid = speaker_ids[i];
			k = 0;
		}
		k++;
		// ������������׼��
		SpeakerSample feature = loadFeatureFile(feature_paths[i], speaker_ids[i]);
		std::cout << "Speaker id:" << speaker_ids[i] << " ��" << k << "�������ļ�" << std::endl;
		feature.features = normalize_feature(feature.features);
		speaker_features[speaker_ids[i]].push_back(feature.features);
	}

	std::random_device rd;
	std::mt19937 g(rd());

	// Ϊÿ��˵����ѵ��ģ��
	for (const auto& [speaker_id, features] : speaker_features)
	{
		// ����ģ��
		auto model = std::make_shared<ECAPA_TDNN>();
		torch::optim::Adam optimizer(
			model->parameters(),
			torch::optim::AdamOptions(1e-3)
		);

		// ������ʧ����
		SpeakerVerificationLoss verification_loss;

		// ѵ������
		int epochs = 100;
		int batch_size = 8;

		for (int epoch = 0; epoch < epochs; ++epoch)
		{
			float total_loss = 0.0;

			// �����������
			//std::shuffle(features.begin(), features.end(), g);

			// ������ѵ��
			for (size_t batch_start = 0; batch_start < features.size(); batch_start += batch_size)
			{
				// ׼����������
				std::vector<torch::Tensor> batch_features(
					features.begin() + batch_start,
					features.begin() + std::min(batch_start + batch_size, features.size())
				);

				// ����Ƕ��
				std::vector<torch::Tensor> embeddings;
				for (auto& feat : batch_features)
				{
					embeddings.push_back(model->forward(feat.unsqueeze(0)));
				}

				// ������������
				torch::Tensor anchor = embeddings[0];
				torch::Tensor positives = torch::stack(
					std::vector<torch::Tensor>(
						embeddings.begin() + 1,
						embeddings.end()
					)
				);

				// ������˵���˻�ȡ������
				torch::Tensor negatives = sample_negatives(speaker_features, speaker_id);

				// ������ʧ
				optimizer.zero_grad();
				torch::Tensor loss = verification_loss.forward(anchor, positives, negatives);
				loss.backward();
				optimizer.step();

				total_loss += loss.item<float>();
			}
			int p_speaker_id = speaker_id;
			float lossAvg = total_loss / (features.size() / batch_size);
			// ��ӡÿ����ʧ
			if (epoch % 10 == 0)
			{
				std::cout << "Speaker " << p_speaker_id << ", Epoch " << epoch << ", Avg Loss: " << lossAvg << std::endl;
			}
		}

		// ����ģ��
		std::string model_path = output_dir + "/speaker_" + std::to_string(speaker_id) + "_model.pt";
		torch::save(model, model_path);
	}
}

std::vector<std::string> SpeakerDataSet::getFeaturePath() const
{
	return feature_paths;
}

std::vector<int> SpeakerDataSet::getSpeakerId() const
{
	return speaker_ids;
}

torch::Tensor SpeakerDataSet::vectorToTensor(std::vector<std::vector<float>> feature_sample)
{
	size_t rows = feature_sample.size();
	size_t cols = feature_sample.at(0).size();

	// ����һ��Tensor�ڴ�����
	torch::Tensor tensor = torch::zeros({ static_cast<long>(rows), static_cast<long>(cols) },
		torch::TensorOptions().dtype(torch::kFloat32));

	// ��������
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			tensor[i][j] = feature_sample.at(i).at(j);
		}
	}

	return tensor;
}

torch::Tensor SpeakerDataSet::sample_negatives(const std::unordered_map<int, std::vector<torch::Tensor>>& all_features, int current_speaker_id, int num_negatives)
{
	std::vector<torch::Tensor> negative_samples;

	std::random_device rd;
	std::mt19937 g(rd());

	// �ռ����зǵ�ǰ˵���˵�����
	std::vector<torch::Tensor> other_features;
	for (const auto& [speaker_id, features] : all_features)
	{
		if (speaker_id != current_speaker_id)
		{
			other_features.insert(
				other_features.end(),
				features.begin(),
				features.end()
			);
		}
	}

	// �������
	//std::shuffle(other_features.begin(), other_features.end(), g);
	for (int i = 0; i < std::min(num_negatives, (int)other_features.size()); ++i)
	{
		negative_samples.push_back(other_features[i]);
	}

	return torch::stack(negative_samples);
}

torch::Tensor SpeakerDataSet::normalize_feature(torch::Tensor feature)
{
	return (feature - feature.mean()) / feature.std();
}