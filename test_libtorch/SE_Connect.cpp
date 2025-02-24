#include "SE_Connect.h"

SE_Connect::SE_Connect(int64_t channels, int64_t s)
{
	TORCH_CHECK(channels % s == 0, channels, " % ", s, " != 0");
	linear1 = this->register_module("linear1", torch::nn::Linear(channels, channels / s));
	linear2 = this->register_module("linear2", torch::nn::Linear(channels / s, channels));
}

torch::Tensor SE_Connect::forward(torch::Tensor x)
{
	auto out = x.mean(2);
	out = torch::relu(linear1->forward(out));
	out = torch::sigmoid(linear2->forward(out));
	return x * out.unsqueeze(2);
}