#include<opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

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
		std::string image_path = "M:\\fd\\ECAPA_DTNN_VPT\\pyfftcomf\\flower.jpg";  // ʹ����ȷ�Ͽ��Զ�ȡ��·��
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
		std::string model_path = "M:\\fd\\ECAPA_DTNN_VPT\\pyfftcomf\\resnet34.pt";
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

int main()
{
	cuda_test();
	return 0;
}