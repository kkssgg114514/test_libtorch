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
    //将wav数据大批量转化为feature数据,也可以实现小批量
    //该函数会递归搜索输入目录下的所有wav文件
    void processBatchMfccFeatures(const std::string& input_dir, const std::string& output_dir);

    //提取特征文件,保存为样本(单)
    SpeakerSample loadFeatureFile(const std::string& input_path, int64_t id);

    //从目录文件读取数据
    void load_feature_paths(const std::string& path_file);

    //训练说话人模型
    void train_speaker_models(const std::string& output_dir);

    std::vector<std::string> getFeaturePath() const;

    std::vector<int> getSpeakerId() const;

private:
    //提取的feature转换为tensor
    torch::Tensor vectorToTensor(std::vector<std::vector<float>> feature_sample);

    //提取负样本
    std::vector<torch::Tensor> sample_negatives(
        const std::unordered_map<int, std::vector<torch::Tensor>>& all_features,
        int current_speaker_id,
        int num_negatives = 5
    );

    //向量标准化
    torch::Tensor normalize_feature(torch::Tensor feature);

    std::vector<std::string> feature_paths;
    std::vector<int> speaker_ids;
};

