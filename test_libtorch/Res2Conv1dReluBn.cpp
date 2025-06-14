#include "Res2Conv1dReluBn.h"
#include <vector>

Res2Conv1dReluBn::Res2Conv1dReluBn(int64_t channels, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool bias, int64_t scale)
	: scale_(scale), width_(channels / scale), nums_(scale == 1 ? scale : scale - 1)
{
	TORCH_CHECK(channels % scale == 0, channels, " % ", scale, " != 0");

	for (int i = 0; i < nums_; i++)
	{
		convs_->push_back(torch::nn::Conv1d(
			torch::nn::Conv1dOptions(width_, width_, kernel_size)
			.stride(stride)
			.padding(padding)
			.dilation(dilation)
			.bias(bias)));
		bns_->push_back(torch::nn::BatchNorm1d(width_));
	}
	convs_ = this->register_module("convs", convs_);
	bns_ = this->register_module("bns", bns_);
}

torch::Tensor Res2Conv1dReluBn::forward(torch::Tensor x)
{
	std::vector<torch::Tensor> spx = torch::split(x, width_, 1);
	std::vector<torch::Tensor> out;
	torch::Tensor sp;

	for (int i = 0; i < nums_; i++)
	{
		if (i == 0)
		{
			sp = spx[i];
		}
		else
		{
			sp = sp + spx[i];
		}
		auto conv = std::dynamic_pointer_cast<torch::nn::Conv1dImpl>(convs_->ptr(i));  // 获取具体的模块
		auto bn = std::dynamic_pointer_cast<torch::nn::BatchNorm1dImpl>(bns_->ptr(i));
		sp = conv->forward(sp);  // 显式调用forward方法
		sp = bn->forward(torch::relu(sp));
		out.push_back(sp);
	}

	if (scale_ != 1)
	{
		out.push_back(spx[nums_]);
	}

	return torch::cat(out, 1);
}