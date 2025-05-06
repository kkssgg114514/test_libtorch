#include "SpeakerVerificationLoss.h"

SpeakerVerificationLoss::SpeakerVerificationLoss(float margin)
	: margin_(margin)
{
}

torch::Tensor SpeakerVerificationLoss::forward(torch::Tensor anchor, torch::Tensor positives, torch::Tensor negatives)
{
   /* std::cout << anchor.sizes() << std::endl;
	 std::cout << positives.sizes() << std::endl;
	 std::cout << negatives.sizes() << std::endl;*/
	// ����ê���������������������ƶ�
	auto pos_similarity = torch::cosine_similarity(anchor.unsqueeze(0), positives);
	auto neg_similarity = torch::cosine_similarity(anchor.unsqueeze(0), negatives);

	// �Ա���ʧ�����������ƶ�Ӧ�ӽ�1�����������ƶ�ӦԶ��1
	auto pos_loss = torch::clamp(1.0 - pos_similarity, 0.0);
	auto neg_loss = torch::clamp(neg_similarity - margin_, 0.0);
	/*std::cout << pos_loss.sizes() << std::endl;
	std::cout << neg_loss.sizes() << std::endl;*/
	//��������

	return torch::mean(pos_loss + neg_loss);
}