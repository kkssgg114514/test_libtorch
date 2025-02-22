#pragma once
#include <torch/torch.h>
#include "SE_Res2Block.h"
#include "AttentiveStatsPool.h"

class ECAPA_TDNN : public torch::nn::Module
{
public:
	ECAPA_TDNN(int64_t in_channels = 80, int64_t channels = 512, int64_t embd_dim = 192);

	torch::Tensor forward(torch::Tensor x);

private:
	std::shared_ptr<Conv1dReluBn> layer1;
	std::shared_ptr<SE_Res2Block> layer2;
	std::shared_ptr<SE_Res2Block> layer3;
	std::shared_ptr<SE_Res2Block> layer4;
	torch::nn::Conv1d conv { nullptr };
	std::shared_ptr<AttentiveStatsPool> pooling;
	torch::nn::BatchNorm1d bn1 { nullptr };
	torch::nn::Linear linear { nullptr };
	torch::nn::BatchNorm1d bn2 { nullptr };
};
