#include "SpeakerDataSet.h"
#include "ECAPA_TDNN.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "SpeakerVerificationLoss.h"
#include <random>
#include <filesystem>

void SpeakerDataSet::processBatchMfccFeatures(const std::string& input_dir, const std::string& output_dir)
{
	mfcc_processer.processBatchMfccFeatures(input_dir, output_dir);
}

std::string SpeakerDataSet::processMfccFeatures(const std::string wavPath, const std::string& output_dir)
{
	mfcc_processer.extractMfccFeatures(wavPath, 25, 10);

	// 构建相对路径
	std::filesystem::path input_path(wavPath);

	// 构建对应的输出路径
	std::filesystem::path output_path =
		std::filesystem::path(output_dir) /
		(input_path.stem().string() + ".feature");

	// 确保输出目录存在
	std::filesystem::create_directories(output_path.parent_path());
	// 保存特征
	mfcc_processer.saveMfccFeatures(output_path.string());
	return output_path.string();
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

	while (infile >> feature_path >> speaker_id)
	{
		feature_paths.push_back(feature_path);
		speaker_ids.push_back(speaker_id);
	}
}

void SpeakerDataSet::train_speaker_models(const std::string& output_dir)
{

	// 检查 CUDA 是否可用
	torch::Device device(torch::kCPU);
	if (!torch::cuda::is_available()) {
		std::cout << "CUDA is not available, using CPU instead." << std::endl;
		device = torch::Device(torch::kCPU);
	}

	// 确保输出目录存在
	namespace fs = std::filesystem;
	if (!fs::exists(output_dir)) {
		try {
			fs::create_directories(output_dir);
			std::cout << "Created output directory: " << output_dir << std::endl;
		}
		catch (const fs::filesystem_error& e) {
			std::cerr << "Error creating directory: " << e.what() << std::endl;
			std::cerr << "Cannot proceed without valid output directory" << std::endl;
			return;
		}
	}

	// 按说话人ID分组特征
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
		// 加载特征并标准化
		SpeakerSample feature = loadFeatureFile(feature_paths[i], speaker_ids[i]);
		std::cout << "Speaker id:" << speaker_ids[i] << " 第" << k << "个特征文件" << std::endl;
		feature.features = normalize_feature(feature.features);
		//装进GPU
		feature.features = feature.features.to(device);
		speaker_features[speaker_ids[i]].push_back(feature.features);
	}

	std::random_device rd;
	std::mt19937 g(rd());

	// 为每个说话人训练模型
	for (const auto& [speaker_id, features] : speaker_features)
	{
		// 创建模型
		auto model = std::make_shared<ECAPA_TDNN>();
		model->to(device); // 将模型移动到 GPU 上

		torch::optim::Adam optimizer(
			model->parameters(),
			torch::optim::AdamOptions(0.1)
		);

		// 创建损失函数 - 修改为不需要负样本的损失函数
		// 这里应该使用一个适合单个说话人特征的损失函数，例如MSE或交叉熵
		torch::nn::MSELoss mse_loss;
		mse_loss->to(device);

		// 训练参数
		int epochs = 100;
		int batch_size = 8;
		std::vector<torch::Tensor> all_embeddings;
		for (int epoch = 0; epoch < epochs; ++epoch)
		{
			float total_loss = 0.0;
			model->train();

			// 输出当前学习率
			float lr = optimizer.param_groups()[0].options().get_lr();
			std::cout << "Speaker " << speaker_id << ", Epoch " << epoch << ", Learning Rate: " << lr << std::endl;
			
			// 分批次训练
			for (size_t batch_start = 0; batch_start < features.size(); batch_start += batch_size)
			{
				// 准备批次数据
				std::vector<torch::Tensor> batch_features(
					features.begin() + batch_start,
					features.begin() + std::min(batch_start + batch_size, features.size())
				);

				// 将批次数据移动到 GPU 上
				for (auto& feat : batch_features)
				{
					if (feat.device() != device)
					{
						feat = feat.to(device);
					}
				}

				// 生成嵌入
				std::vector<torch::Tensor> embeddings;
				for (auto& feat : batch_features)
				{
					auto input = feat.unsqueeze(0).expand({ 2, -1, -1 }).to(device);
					auto emb = model->forward(input).to(device);
					embeddings.push_back(emb);
					all_embeddings.push_back(emb);
				}

				if (embeddings.size() <= 1) {
					continue; // 跳过只有一个样本的批次
				}

				// 构建自监督学习任务：预测同一说话人的其他特征
				torch::Tensor anchor = embeddings[0].to(device);
				torch::Tensor target = torch::stack(
					std::vector<torch::Tensor>(
						embeddings.begin() + 1,
						embeddings.end()
					)
				).to(device);

				// 计算损失 - 使用均方误差或其他适合的损失函数
				// 这里假设我们希望同一说话人的不同特征产生相似的嵌入表示
				optimizer.zero_grad();

				// 计算锚嵌入与所有其他嵌入之间的距离
				torch::Tensor expanded_anchor = anchor.expand_as(target).to(device);
				torch::Tensor loss = mse_loss(expanded_anchor, target).to(device);

				loss.backward();
				optimizer.step();

				total_loss += loss.detach().item<float>();
			}

			// 计算梯度范数
			torch::Tensor grad_norm = torch::zeros({ 1 }, torch::kFloat32).to(device);
			for (const auto& param : model->parameters())
			{
				if (param.grad().defined())
				{
					grad_norm += param.grad().norm().pow(2);
				}
			}
			grad_norm = grad_norm.sqrt();

			// 输出梯度范数
			std::cout << "Speaker " << speaker_id << ", Epoch " << epoch << ", Gradient Norm: " << grad_norm.item<float>() << std::endl;

			int p_speaker_id = speaker_id;
			float lossAvg = total_loss / (features.size() / batch_size);
			// 打印每轮损失
			if (epoch % 10 == 0)
			{
				std::cout << "Speaker " << p_speaker_id << ", Epoch " << epoch << ", Avg Loss: " << lossAvg << std::endl;
			}
			
		}

		// 计算平均嵌入作为参考
		torch::Tensor reference_embedding = torch::stack(all_embeddings).mean(0);

		// 保存模型
		std::string model_path = output_dir + "/speaker_" + std::to_string(speaker_id) + "_model.pt";
		torch::save(model, model_path);

		// 保存参考嵌入
		std::string embedding_path = output_dir + "/speaker_" + std::to_string(speaker_id) + "_embedding.pt";
		torch::save(reference_embedding, embedding_path);
	}
}

int SpeakerDataSet::test_speaker_models(const std::string& model_dir, const std::string& test_path)
{
	// 检查 CUDA 是否可用
	torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

	// 加载测试特征
	SpeakerSample test_sample = loadFeatureFile(test_path, 0);
	test_sample.features = normalize_feature(test_sample.features);
	test_sample.features = test_sample.features.to(device);

	// 遍历模型目录，加载所有模型和对应的参考嵌入
	std::vector<std::pair<int, float>> similarities;

	for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
		std::string file_path = entry.path().string();

		// 只处理模型文件
		if (file_path.find("_model.pt") == std::string::npos) {
			continue;
		}

		std::string model_path = file_path;
		int speaker_id = extract_speaker_id_from_path(model_path);

		// 构造对应的嵌入文件路径
		std::string embedding_path = model_dir + "/speaker_" + std::to_string(speaker_id) + "_embedding.pt";

		// 确认嵌入文件存在
		if (!std::filesystem::exists(embedding_path)) {
			std::cerr << "Warning: Reference embedding not found for speaker " << speaker_id << std::endl;
			continue;
		}

		try {
			// 加载特定说话人的模型
			auto speaker_model = std::make_shared<ECAPA_TDNN>();
			torch::load(speaker_model, model_path);
			speaker_model->to(device);
			speaker_model->eval();

			// 加载参考嵌入
			torch::Tensor reference_embedding;
			torch::load(reference_embedding, embedding_path);
			reference_embedding = reference_embedding.to(device);

			// 使用此说话人模型处理测试特征
			torch::NoGradGuard no_grad; // 禁用梯度计算提高效率
			torch::Tensor test_embedding = speaker_model->forward(
				test_sample.features.unsqueeze(0).expand({ 2, -1, -1 })
			);

			// 计算余弦相似度
			float similarity = torch::cosine_similarity(
				test_embedding,
				reference_embedding
			).mean().item<float>();

			similarities.emplace_back(speaker_id, similarity);
			std::cout << "Speaker " << speaker_id << " similarity: " << similarity << std::endl;
		}
		catch (const std::exception& e) {
			std::cerr << "Error processing speaker " << speaker_id << ": " << e.what() << std::endl;
			continue;
		}
	}

	// 找到最相似的说话人
	if (similarities.empty()) {
		std::cerr << "No valid similarities calculated!" << std::endl;
		return -1;
	}

	auto max_similarity = std::max_element(
		similarities.begin(),
		similarities.end(),
		[](const auto& a, const auto& b) { return a.second < b.second; }
	);

	std::cout << "Best match: Speaker " << max_similarity->first
		<< " with similarity " << max_similarity->second << std::endl;

	return max_similarity->first;
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

	// 创建一个Tensor内存区域
	torch::Tensor tensor = torch::zeros({ static_cast<long>(rows), static_cast<long>(cols) },
		torch::TensorOptions().dtype(torch::kFloat32));

	// 复制数据
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			tensor[i][j] = feature_sample.at(i).at(j);
		}
	}

	return tensor;
}

std::vector<torch::Tensor> SpeakerDataSet::sample_negatives(const std::unordered_map<int, std::vector<torch::Tensor>>& all_features, int current_speaker_id, int num_negatives)
{
	std::vector<torch::Tensor> negative_samples;

	std::random_device rd;
	std::mt19937 g(rd());

	// 收集所有非当前说话人的特征
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

	// 随机采样
	std::shuffle(other_features.begin(), other_features.end(), g);
	for (int i = 0; i < num_negatives; ++i)
	{
		negative_samples.push_back(other_features[i]);
	}

	return negative_samples;
}

int SpeakerDataSet::extract_speaker_id_from_path(const std::string& model_path)
{
	size_t firstUnderscore = model_path.find('_');
	if (firstUnderscore == std::string::npos) {
		return -1; // 没找到第一个下划线
	}
	size_t secondUnderscore = model_path.find('_', firstUnderscore + 1);
	if (secondUnderscore == std::string::npos) {
		return -1; // 没找到第二个下划线
	}
	std::string numberStr = model_path.substr(firstUnderscore + 1, secondUnderscore - firstUnderscore - 1);
	try {
		return std::stoi(numberStr);
	}
	catch (const std::invalid_argument& e) {
		std::cerr << "转换数字时出错: " << e.what() << std::endl;
		return -1;
	}
	catch (const std::out_of_range& e) {
		std::cerr << "数字超出范围: " << e.what() << std::endl;
		return -1;
	}
}

torch::Tensor SpeakerDataSet::normalize_feature(torch::Tensor feature)
{
	return (feature - feature.mean()) / feature.std();
}