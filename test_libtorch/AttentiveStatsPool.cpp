#include "AttentiveStatsPool.h"
#include <iostream>

AttentiveStatsPool::AttentiveStatsPool(int64_t in_dim, int64_t bottleneck_dim)
{
	linear1 = this->register_module("linear1", torch::nn::Conv1d(
		torch::nn::Conv1dOptions(in_dim, bottleneck_dim, 1)));
	linear2 = this->register_module("linear2", torch::nn::Conv1d(
		torch::nn::Conv1dOptions(bottleneck_dim, in_dim, 1)));
}

torch::Tensor AttentiveStatsPool::forward(torch::Tensor x)
{
	//std::cout << x.sizes() << std::endl;
	auto alpha = torch::tanh(linear1->forward(x));
	alpha = torch::softmax(linear2->forward(alpha), 2);
	auto mean = torch::sum(alpha * x, 2);
	//std::cout << mean.sizes() << std::endl;
	auto residuals = torch::sum(alpha * x.pow(2), 2) - mean.pow(2);
	auto std = torch::sqrt(residuals.clamp(1e-9));
	//std::cout << mean.sizes() << " " << std.sizes() << std::endl;
	return torch::cat({ mean, std }, 1);
}