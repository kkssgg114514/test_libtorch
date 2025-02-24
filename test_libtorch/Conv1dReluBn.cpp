#include "Conv1dReluBn.h"

Conv1dReluBn::Conv1dReluBn(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool bias)
{
	conv = this->register_module("conv", torch::nn::Conv1d(
		torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
		.stride(stride)
		.padding(padding)
		.dilation(dilation)
		.bias(bias)));
	bn = this->register_module("bn", torch::nn::BatchNorm1d(out_channels));
}

torch::Tensor Conv1dReluBn::forward(torch::Tensor x)
{
	return bn->forward(torch::relu(conv->forward(x)));
}
