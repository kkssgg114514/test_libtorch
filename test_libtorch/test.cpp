//#include <opencv2/opencv.hpp>
#include <iostream>
//
//int main()
//{
//    std::string image_path = "M:\\fd\\ECAPA_DTNN_VPT\\pyfftcomf\\flower.jpg";
//    cv::Mat image = cv::imread(image_path);
//
//    if (image.empty())
//    {
//        std::cout << "Failed to load image at: " << image_path << std::endl;
//        return -1;
//    }
//
//    std::cout << "Successfully loaded image!" << std::endl;
//    std::cout << "Image size: " << image.size() << std::endl;
//    std::cout << "Image channels: " << image.channels() << std::endl;
//
//    return 0;
//}

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <string>

void cpu_test()
{
	try
	{
		// ʹ�� CPU �豸
		auto device = torch::Device(torch::kCPU);

		// ��ȡͼƬ
		std::string image_path = "M:\\fd\\ECAPA_DTNN_VPT\\pyfftcomf\\flower.jpg";
		cv::Mat image = cv::imread(image_path);
		std::cout << "Original image size: " << image.size() << std::endl;

		// ����
		cv::resize(image, image, cv::Size(224, 224));
		std::cout << "Resized image size: " << image.size() << std::endl;

		// ת��Ϊ����
		torch::Tensor input_tensor = torch::from_blob(
			image.data,
			{ 1, image.rows, image.cols, 3 },
			torch::kByte
		);

		// ����ά�Ⱥ���������
		input_tensor = input_tensor.permute({ 0, 3, 1, 2 }).to(torch::kFloat32);
		input_tensor = input_tensor / 255.0;

		// ��ӡ������Ϣ
		std::cout << "Tensor shape: " << input_tensor.sizes() << std::endl;

		// ����ģ��
		std::string model_path = "M:\\fd\\ECAPA_DTNN_VPT\\pyfftcomf\\resnet34.pt";
		auto model = torch::jit::load(model_path);
		model.to(device);
		model.eval();

		// ����
		torch::NoGradGuard no_grad;
		auto output = model.forward({ input_tensor }).toTensor();
		output = torch::softmax(output, 1);

		// ��ȡ���
		auto max_result = output.max(1);
		auto max_index = std::get<1>(max_result).item<int64_t>();
		auto confidence = std::get<0>(max_result).item<float>();

		std::cout << "Ԥ�����: " << max_index << ", ���Ŷ�: " << confidence << std::endl;
	}
	catch (const c10::Error& e)
	{
		std::cout << "PyTorch error: " << e.msg() << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cout << "Standard error: " << e.what() << std::endl;
	}
}

void cuda_test()
{
	try
	{
		// ���CUDA
		if (!torch::cuda::is_available())
		{
			std::cout << "CUDA is not available!" << std::endl;
			return;
		}
		auto device = torch::Device(torch::kCUDA, 0);

		// ��ȡͼƬ
		std::string image_path = "D:\\private\\pycode\\flower.jpg";  // ʹ����ȷ�Ͽ��Զ�ȡ��·��
		cv::Mat image = cv::imread(image_path);
		std::cout << "Original image size: " << image.size() << std::endl;

		// ����
		cv::resize(image, image, cv::Size(224, 224));
		std::cout << "Resized image size: " << image.size() << std::endl;

		// ת��Ϊ����
		torch::Tensor input_tensor = torch::from_blob(
			image.data,
			{ 1, image.rows, image.cols, 3 },
			torch::kByte
		);

		// ����ά�Ⱥ���������
		input_tensor = input_tensor.permute({ 0, 3, 1, 2 }).to(torch::kFloat32);
		input_tensor = input_tensor / 255.0;

		// ��ӡ������Ϣ�Լ��
		std::cout << "Tensor shape: " << input_tensor.sizes() << std::endl;
		std::cout << "Tensor device: " << input_tensor.device() << std::endl;

		// ����ģ��
		std::string model_path = "D:\\private\\pycode\\resnet34.pt";
		auto model = torch::jit::load(model_path);
		model.to(device);
		model.eval();

		// �������Ƶ�GPU
		input_tensor = input_tensor.to(device);

		// ����
		torch::NoGradGuard no_grad;
		auto output = model.forward({ input_tensor }).toTensor();
		output = torch::softmax(output, 1);

		// ��ȡ���
		auto max_result = output.max(1);
		auto max_index = std::get<1>(max_result).item<int64_t>();
		auto confidence = std::get<0>(max_result).item<float>();

		std::cout << "Ԥ�����: " << max_index << ", ���Ŷ�: " << confidence << std::endl;
	}
	catch (const c10::Error& e)
	{
		std::cout << "PyTorch error: " << e.msg() << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cout << "Standard error: " << e.what() << std::endl;
	}
}

void org_cuda()
{
	//����ʹ��cuda
	auto device = torch::Device(torch::kCUDA, 0);
	//��ȡͼƬ
	auto image = cv::imread("M:\\fd\\ECAPA_DTNN_VPT\\pyfftcomf\\flower.jpg");
	//������ָ����С
	cv::resize(image, image, cv::Size(224, 224));
	//ת������
	auto input_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat32) / 225.0;
	//����ģ��
	auto model = torch::jit::load("M:\\fd\\ECAPA_DTNN_VPT\\pyfftcomf\\resnet34.pt");
	model.to(device);
	model.eval();
	//ǰ�򴫲�
	auto output = model.forward({ input_tensor.to(device) }).toTensor();
	output = torch::softmax(output, 1);
	std::cout << "ģ��Ԥ����Ϊ��" << torch::argmax(output) << "�࣬���Ŷ�Ϊ" << output.max() << std::endl;;
}

//int main()
//{
//	cuda_test();
//	return 0;
//}

#include "Filesearch.h"
//
//int main()
//{
//	std::vector<std::string> testf = Filesearch::getAllFiles("D:/private/fd/data_aishell/wav");
//
//	for (auto i : testf)
//	{
//		std::cout << i << std::endl;
//	}
//}

#include "SpeakerDataSet.h"

//int main()
//{
//	VPT_Mfcc mfcc_processor;
//
//	//// ����Ŀ¼�����Ŀ¼
//	//std::string input_dir = "D:/private/fd/data_aishell/wav";
//	//std::string output_dir = "D:/private/fd/testfeature";
//
//	//// ��������
//	//mfcc_processor.processBatchMfccFeatures(input_dir, output_dir);
//
//	/*mfcc_processor.loadMfccFeatures("D:/private/fd/testfeature/S0002/train/S0002/BAC009S0002W0122.feature");
//
//	std::vector<std::vector<float>> feature_sample = mfcc_processor.getFeature();
//
//	for (auto& i : feature_sample)
//	{
//		for (auto& j : i)
//		{
//			std::cout << j << " ";
//		}
//		std::cout << std::endl;
//	}*/
//
//	return 0;
//}


int main()
{
	SpeakerDataSet test;
	test.load_feature_paths("D:\\private\\fd\\filedir.txt");

	std::vector<std::string> featurePath = test.getFeaturePath();
	std::vector<int> speakerId = test.getSpeakerId();

	for (auto& i : featurePath)
	{
		std::cout << i << std::endl;
	}


	return 0;
}