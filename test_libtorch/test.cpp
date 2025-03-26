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
		// 使用 CPU 设备
		auto device = torch::Device(torch::kCPU);

		// 读取图片
		std::string image_path = "M:\\fd\\ECAPA_DTNN_VPT\\pyfftcomf\\flower.jpg";
		cv::Mat image = cv::imread(image_path);
		std::cout << "Original image size: " << image.size() << std::endl;

		// 缩放
		cv::resize(image, image, cv::Size(224, 224));
		std::cout << "Resized image size: " << image.size() << std::endl;

		// 转换为张量
		torch::Tensor input_tensor = torch::from_blob(
			image.data,
			{ 1, image.rows, image.cols, 3 },
			torch::kByte
		);

		// 调整维度和数据类型
		input_tensor = input_tensor.permute({ 0, 3, 1, 2 }).to(torch::kFloat32);
		input_tensor = input_tensor / 255.0;

		// 打印张量信息
		std::cout << "Tensor shape: " << input_tensor.sizes() << std::endl;

		// 加载模型
		std::string model_path = "M:\\fd\\ECAPA_DTNN_VPT\\pyfftcomf\\resnet34.pt";
		auto model = torch::jit::load(model_path);
		model.to(device);
		model.eval();

		// 推理
		torch::NoGradGuard no_grad;
		auto output = model.forward({ input_tensor }).toTensor();
		output = torch::softmax(output, 1);

		// 获取结果
		auto max_result = output.max(1);
		auto max_index = std::get<1>(max_result).item<int64_t>();
		auto confidence = std::get<0>(max_result).item<float>();

		std::cout << "预测类别: " << max_index << ", 置信度: " << confidence << std::endl;
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
		// 检查CUDA
		if (!torch::cuda::is_available())
		{
			std::cout << "CUDA is not available!" << std::endl;
			return;
		}
		auto device = torch::Device(torch::kCUDA, 0);

		// 读取图片
		std::string image_path = "D:\\private\\pycode\\flower.jpg";  // 使用已确认可以读取的路径
		cv::Mat image = cv::imread(image_path);
		std::cout << "Original image size: " << image.size() << std::endl;

		// 缩放
		cv::resize(image, image, cv::Size(224, 224));
		std::cout << "Resized image size: " << image.size() << std::endl;

		// 转换为张量
		torch::Tensor input_tensor = torch::from_blob(
			image.data,
			{ 1, image.rows, image.cols, 3 },
			torch::kByte
		);

		// 调整维度和数据类型
		input_tensor = input_tensor.permute({ 0, 3, 1, 2 }).to(torch::kFloat32);
		input_tensor = input_tensor / 255.0;

		// 打印张量信息以检查
		std::cout << "Tensor shape: " << input_tensor.sizes() << std::endl;
		std::cout << "Tensor device: " << input_tensor.device() << std::endl;

		// 加载模型
		std::string model_path = "D:\\private\\pycode\\resnet34.pt";
		auto model = torch::jit::load(model_path);
		model.to(device);
		model.eval();

		// 将张量移到GPU
		input_tensor = input_tensor.to(device);

		// 推理
		torch::NoGradGuard no_grad;
		auto output = model.forward({ input_tensor }).toTensor();
		output = torch::softmax(output, 1);

		// 获取结果
		auto max_result = output.max(1);
		auto max_index = std::get<1>(max_result).item<int64_t>();
		auto confidence = std::get<0>(max_result).item<float>();

		std::cout << "预测类别: " << max_index << ", 置信度: " << confidence << std::endl;
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
	//定义使用cuda
	auto device = torch::Device(torch::kCUDA, 0);
	//读取图片
	auto image = cv::imread("M:\\fd\\ECAPA_DTNN_VPT\\pyfftcomf\\flower.jpg");
	//缩放至指定大小
	cv::resize(image, image, cv::Size(224, 224));
	//转成张量
	auto input_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat32) / 225.0;
	//加载模型
	auto model = torch::jit::load("M:\\fd\\ECAPA_DTNN_VPT\\pyfftcomf\\resnet34.pt");
	model.to(device);
	model.eval();
	//前向传播
	auto output = model.forward({ input_tensor.to(device) }).toTensor();
	output = torch::softmax(output, 1);
	std::cout << "模型预测结果为第" << torch::argmax(output) << "类，置信度为" << output.max() << std::endl;;
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
//	//// 输入目录和输出目录
//	//std::string input_dir = "D:/private/fd/data_aishell/wav";
//	//std::string output_dir = "D:/private/fd/testfeature";
//
//	//// 批量处理
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