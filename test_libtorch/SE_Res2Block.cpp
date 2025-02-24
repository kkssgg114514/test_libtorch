#include "SE_Res2Block.h"

SE_Res2Block::SE_Res2Block(int64_t channels, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, int64_t scale)
{
	conv1 = this->register_module("conv1", std::make_shared<Conv1dReluBn>(channels, channels, 1, 1, 0));
	res2conv = this->register_module("res2conv", std::make_shared<Res2Conv1dReluBn>(
		channels, kernel_size, stride, padding, dilation, false, scale));
	conv2 = this->register_module("conv2", std::make_shared<Conv1dReluBn>(channels, channels, 1, 1, 0));
	se = this->register_module("se", std::make_shared<SE_Connect>(channels));
}

torch::Tensor SE_Res2Block::forward(torch::Tensor x)
{
	auto out = conv1->forward(x);
	out = res2conv->forward(out);
	out = conv2->forward(out);
	return se->forward(out);
}
