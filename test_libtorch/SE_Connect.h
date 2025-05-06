#pragma once
#include <torch/torch.h>

class SE_Connect : public torch::nn::Module
{
public:
	SE_Connect(int64_t channels, int64_t s = 2);

	torch::Tensor forward(torch::Tensor x);

private:
	torch::nn::Linear linear1 { nullptr };
	torch::nn::Linear linear2 { nullptr };
};
