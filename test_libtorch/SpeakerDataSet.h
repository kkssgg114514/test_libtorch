#pragma once
#include <torch/torch.h>


struct SpeakerSample {
    torch::Tensor features;  // MFCC���� [seq_len, 80]
    int64_t speaker_id;      // ˵����ID
};

class SpeakerDataSet : public torch::data::Dataset<SpeakerDataSet>
{
private:
    struct SpeakerData
    {
        std::string feature_path;
        int64_t speaker_id;
    };

    std::vector<SpeakerData> samples;
    int feature_dim;
    int chunk_size;
    bool is_train;

public:
    SpeakerDataSet(const std::string& feature_dir, int feature_dim = 80,
        int chunk_size = 200, bool is_train = true);

    //��ȡ˵��������
    int64_t num_speakers() const;

    torch::optional<size_t> size() const override;

    SpeakerSample get(size_t index) override;
};

