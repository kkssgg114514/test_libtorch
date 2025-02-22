#pragma once
#include <torch/torch.h>

class AttentiveStatsPool
{
public:
	AttentiveStatsPool(int64_t in_dim, int64_t bottleneck_dim);

	torch::Tensor forward(torch::Tensor x);

private:
	torch::nn::Conv1d linear1 { nullptr };
	torch::nn::Conv1d linear2 { nullptr };
};

