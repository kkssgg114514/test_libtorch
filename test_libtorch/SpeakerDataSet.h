#pragma once
#include <torch/torch.h>
#include "VPT_Mfcc.h"

struct SpeakerSample {
    torch::Tensor features;  // MFCC���� [seq_len, 80]
    int64_t speaker_id;      // ˵����ID
};

class SpeakerDataSet
{
private:
    VPT_Mfcc mfcc_processer;

public:
    //��wav���ݴ�����ת��Ϊfeature����,Ҳ����ʵ��С����
    //�ú�����ݹ���������Ŀ¼�µ�����wav�ļ�
    void processBatchMfccFeatures(const std::string& input_dir, const std::string& output_dir);

    //��ȡ�����ļ�,����Ϊ����(��)
    SpeakerSample loadFeatureFile(const std::string& input_path, int64_t id);

    //��Ŀ¼�ļ���ȡ����
    void load_feature_paths(const std::string& path_file);

private:
    //��ȡ��featureת��Ϊtensor
    torch::Tensor vectorToTensor(std::vector<std::vector<float>> feature_sample);


    std::vector<std::string> feature_paths;
    std::vector<int> speaker_ids;
};

