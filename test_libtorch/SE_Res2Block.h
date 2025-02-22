#pragma once
#include <torch/torch.h>
#include "Res2Conv1dReluBn.h"
#include "Conv1dReluBn.h"
#include "SE_Connect.h"

class SE_Res2Block : public torch::nn::Module
{
public:
	SE_Res2Block(int64_t channels, int64_t kernel_size, int64_t stride,
				 int64_t padding, int64_t dilation, int64_t scale);

	torch::Tensor forward(torch::Tensor x);

private:
	std::shared_ptr<Conv1dReluBn> conv1;
	std::shared_ptr<Res2Conv1dReluBn> res2conv;
	std::shared_ptr<Conv1dReluBn> conv2;
	std::shared_ptr<SE_Connect> se;
};
