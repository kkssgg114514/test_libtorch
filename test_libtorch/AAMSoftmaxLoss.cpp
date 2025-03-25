#include "AAMSoftmaxLoss.h"

AAMSoftmaxLoss::AAMSoftmaxLoss(int64_t embedding_size, int64_t num_classes, float margin, float scale)
	: margin(margin), scale(scale)
{
	embedding = register_module("embedding", torch::nn::Linear(embedding_size, num_classes, false));
	torch::nn::init::xavier_normal_(embedding->weight);
}

std::tuple<torch::Tensor, torch::Tensor> AAMSoftmaxLoss::forward(const torch::Tensor& features, const torch::Tensor& labels)
{
    // ��һ��Ȩ��
    auto weight = torch::nn::functional::normalize(embedding->weight,
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    // ��һ������
    auto normalized_features = torch::nn::functional::normalize(features,
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    // �����������ƶ�
    auto cosine = torch::matmul(normalized_features, weight.transpose(0, 1));

    // �ҳ���ȷ��������ֵ
    auto one_hot = torch::zeros_like(cosine);
    one_hot.scatter_(1, labels.view({ -1, 1 }), 1.0);

    // Ӧ��AAM margin
    auto arc_cosine = cosine - one_hot * margin;

    // Ӧ��scale
    auto logits = arc_cosine * scale;

    // ���㽻������ʧ
    auto loss = torch::nn::functional::cross_entropy(logits, labels);

    return std::make_tuple(loss, cosine);
}

