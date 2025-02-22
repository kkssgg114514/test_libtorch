#pragma once
#include <torch/torch.h>


class Res2Conv1dReluBn : public torch::nn::Module
{
public:
	Res2Conv1dReluBn(int64_t channels, int64_t kernel_size = 1, int64_t stride = 1,
					 int64_t padding = 0, int64_t dilation = 1, bool bias = false, int64_t scale = 4);

	//Ç°Ïò´«²¥
	torch::Tensor forward(torch::Tensor x);

private:
	int64_t scale_;
	int64_t width_;
	int64_t nums_;
	torch::nn::ModuleList convs_;
	torch::nn::ModuleList bns_;
};
