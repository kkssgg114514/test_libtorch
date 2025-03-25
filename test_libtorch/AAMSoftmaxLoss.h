#pragma once
#include <torch/torch.h>

class AAMSoftmaxLoss : public torch::nn::Module
{
private:
    torch::nn::Linear embedding{ nullptr };
    float margin;
    float scale;

public:
    AAMSoftmaxLoss(int64_t embedding_size, int64_t num_classes, float margin = 0.2, float scale = 30.0);

    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& features, const torch::Tensor& labels);
};

