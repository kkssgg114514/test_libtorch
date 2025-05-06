#pragma once
#include <torch/torch.h>
class SpeakerVerificationLoss : torch::nn::Module
{
public:
	SpeakerVerificationLoss(float margin = 0.2);

	torch::Tensor forward(torch::Tensor anchor,
						  torch::Tensor positives,
						  torch::Tensor negatives);

private:
	float margin_;
};
