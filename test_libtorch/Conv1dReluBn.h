#pragma once
#include <torch/torch.h>

class Conv1dReluBn : public torch::nn::Module
{
public:
	Conv1dReluBn(int64_t in_channels, int64_t out_channels, int64_t kernel_size = 1,
				 int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1, bool bias = false);

	torch::Tensor forward(torch::Tensor x);

private:
	torch::nn::Conv1d conv { nullptr };
	torch::nn::BatchNorm1d bn { nullptr };
};

