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
    //ĳ����Ƶ�ļ�ת��Ϊfeature
    std::string processMfccFeatures(const std::string wavPath, const std::string& output_dir);

    //��ȡ�����ļ�,����Ϊ����(��)
    SpeakerSample loadFeatureFile(const std::string& input_path, int64_t id);

    //ѵ��˵����ģ��
    void train_speaker_models(const std::string& output_dir);

	// ѵ��˵����ģ��(��)
	void train_speaker_models_2(const std::string& output_dir);

    //ģ�Ͳ���
    int test_speaker_models(const std::string& model_dir, const std::string& test_path);

    int test_speaker_models_2(const std::string& model_dir, const std::string& test_path);

    std::vector<std::string> getFeaturePath() const;

    std::vector<int> getSpeakerId() const;

	void setFeaturePath(const std::vector<std::string>& feature_path);

	void setSpeakerId(const std::vector<int>& speaker_id);

private:
    //��ȡ��featureת��Ϊtensor
    torch::Tensor vectorToTensor(std::vector<std::vector<float>> feature_sample);

    //��ȡ������
    std::vector<torch::Tensor> sample_negatives(
        const std::unordered_map<int, std::vector<torch::Tensor>>& all_features,
        int current_speaker_id,
        int num_negatives = 5
    );

    // ��ȡ˵����ID,��ģ������
    int extract_speaker_id_from_path(const std::string& model_path);

    //������׼��
    torch::Tensor normalize_feature(torch::Tensor feature);

    std::vector<std::string> feature_paths;
    std::vector<int> f_speaker_ids;

    // ���ڴ洢��Ƶ·���Ͷ�Ӧ�� ID
    std::vector<std::pair<std::string, int>> audio_data;

public:
    //������Ƶ�ļ���Ŀ¼,�Զ�����
	void save_audio_to_directory(const std::string& wavDir);

    //��ȡ��Ƶ�ļ�Ŀ¼
    void load_wav_index();

    //��wav���ݴ�����ת��Ϊfeature����,Ҳ����ʵ��С����
    //�ú�����ݹ���������Ŀ¼�µ�����wav�ļ�
    void processBatchMfccFeatures(const std::string& input_dir, const std::string& output_dir);

    //�������е���Ƶ�ļ�,ʹ�����������ļ�
    void processBatchWavMfccFeatures();

    //��Ŀ¼�ļ���ȡ����
    void load_feature_paths(const std::string& path_file);

    void train_speaker_models_3(const std::string& output_dir);

    int test_speaker_models_3(const std::string& test_path);


private:
    std::vector<int> speaker_ids;
    std::vector<std::string> wav_paths;

    // ����ļ�·�������ڱ�����Ƶ·���Ͷ�Ӧ�� ID
    std::string output_file = "..\\audio_paths.txt";
    std::string feature_path_file = "..\\feaDir.txt";
};

