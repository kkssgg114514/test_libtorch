#include "ECAPA_TDNN.h"

ECAPA_TDNN::ECAPA_TDNN(int64_t in_channels, int64_t channels, int64_t embd_dim)
{
	layer1 = this->register_module("layer1", std::make_shared<Conv1dReluBn>(
		in_channels, channels, 5, 1, 2));

	layer2 = this->register_module("layer2", std::make_shared<SE_Res2Block>(
		channels, 3, 1, 2, 2, 8));

	layer3 = this->register_module("layer3", std::make_shared<SE_Res2Block>(
		channels, 3, 1, 3, 3, 8));

	layer4 = this->register_module("layer4", std::make_shared<SE_Res2Block>(
		channels, 3, 1, 4, 4, 8));

	conv = this->register_module("conv", torch::nn::Conv1d(
		torch::nn::Conv1dOptions(channels * 3, 1536, 1)));

	pooling = this->register_module("pooling", std::make_shared<AttentiveStatsPool>(1536, 128));
	bn1 = this->register_module("bn1", torch::nn::BatchNorm1d(3072));
	linear = this->register_module("linear", torch::nn::Linear(3072, embd_dim));
	bn2 = this->register_module("bn2", torch::nn::BatchNorm1d(embd_dim));
}

torch::Tensor ECAPA_TDNN::forward(torch::Tensor x)
{
	x = x.transpose(1, 2);
	auto out1 = layer1->forward(x);
	auto out2 = layer2->forward(out1) + out1;
	auto out3 = layer3->forward(out1 + out2) + out1 + out2;
	auto out4 = layer4->forward(out1 + out2 + out3) + out1 + out2 + out3;

	auto out = torch::cat({ out2, out3, out4 }, 1);
	out = torch::relu(conv->forward(out));
	out = bn1->forward(pooling->forward(out));
	out = bn2->forward(linear->forward(out));
	return out;
}