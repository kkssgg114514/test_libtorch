#pragma once
#include <torch/torch.h>
#include "VPT_Mfcc.h"

struct SpeakerSample {
    torch::Tensor features;  // MFCC特征 [seq_len, 80]
    int64_t speaker_id;      // 说话人ID
};

class SpeakerDataSet
{
private:
    VPT_Mfcc mfcc_processer;

public:
    //某个音频文件转变为feature
    std::string processMfccFeatures(const std::string wavPath, const std::string& output_dir);

    //提取特征文件,保存为样本(单)
    SpeakerSample loadFeatureFile(const std::string& input_path, int64_t id);

    //训练说话人模型
    void train_speaker_models(const std::string& output_dir);

	// 训练说话人模型(新)
	void train_speaker_models_2(const std::string& output_dir);

    //模型测试
    int test_speaker_models(const std::string& model_dir, const std::string& test_path);

    int test_speaker_models_2(const std::string& model_dir, const std::string& test_path);

    std::vector<std::string> getFeaturePath() const;

    std::vector<int> getSpeakerId() const;

	void setFeaturePath(const std::vector<std::string>& feature_path);

	void setSpeakerId(const std::vector<int>& speaker_id);

private:
    //提取的feature转换为tensor
    torch::Tensor vectorToTensor(std::vector<std::vector<float>> feature_sample);

    //提取负样本
    std::vector<torch::Tensor> sample_negatives(
        const std::unordered_map<int, std::vector<torch::Tensor>>& all_features,
        int current_speaker_id,
        int num_negatives = 5
    );

    // 提取说话人ID,从模型名称
    int extract_speaker_id_from_path(const std::string& model_path);

    //向量标准化
    torch::Tensor normalize_feature(torch::Tensor feature);

    std::vector<std::string> feature_paths;
    std::vector<int> f_speaker_ids;

    // 用于存储音频路径和对应的 ID
    std::vector<std::pair<std::string, int>> audio_data;

public:
    //保存音频文件到目录,自动分组
	void save_audio_to_directory(const std::string& wavDir);

    //读取音频文件目录
    void load_wav_index();

    //将wav数据大批量转化为feature数据,也可以实现小批量
    //该函数会递归搜索输入目录下的所有wav文件
    void processBatchMfccFeatures(const std::string& input_dir, const std::string& output_dir);

    //处理所有的音频文件,使其生成特征文件
    void processBatchWavMfccFeatures();

    //从目录文件读取数据
    void load_feature_paths(const std::string& path_file);

    void train_speaker_models_3(const std::string& output_dir);

    int test_speaker_models_3(const std::string& test_path);


private:
    std::vector<int> speaker_ids;
    std::vector<std::string> wav_paths;

    // 输出文件路径，用于保存音频路径和对应的 ID
    std::string output_file = "..\\audio_paths.txt";
    std::string feature_path_file = "..\\feaDir.txt";
};

