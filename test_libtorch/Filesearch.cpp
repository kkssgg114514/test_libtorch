#include "Filesearch.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <algorithm> // ��Ӵ�ͷ�ļ���ʹ�� std::equal �� std::transform

namespace fs = std::filesystem;

std::vector<std::string> Filesearch::getAllFiles(const std::string& url)
{
	std::vector<std::string> wavFiles;

	try {
		// ���Ŀ¼�Ƿ����
		if (!std::filesystem::exists(url)) {
			std::cerr << "Ŀ¼������: " << url << std::endl;
			return wavFiles;
		}

		// �ݹ����Ŀ¼�е������ļ�����Ŀ¼
		for (const auto& entry : std::filesystem::recursive_directory_iterator(url)) {
			// ����Ƿ�Ϊ��ͨ�ļ�
			if (entry.is_regular_file()) {
				std::string extension = entry.path().extension().string();
				// ת��ΪСд���бȽ�
				std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

				if (extension == ".wav") {
					// ��ȡ�����ľ���·��
					wavFiles.push_back(std::filesystem::absolute(entry.path()).string());
				}
			}
		}
	}
	catch (const std::filesystem::filesystem_error& e) {
		std::cerr << "�ļ�ϵͳ����: " << e.what() << std::endl;
	}

	return wavFiles;
}

void Filesearch::generate_feature_paths(const std::string& root_dir, const std::string& output_file, int speaker_id)
{
	// ���� speaker_id �����ļ���ģʽ
	std::ios_base::openmode mode = (speaker_id == 1) ? (std::ios::out | std::ios::trunc) : (std::ios::out | std::ios::app);
	// ���ļ�
	std::ofstream outfile(output_file, mode);
	if (!outfile.is_open()) {
		std::cerr << "Error: Could not open output file: " << output_file << std::endl;
		return;
	}

	namespace fs = std::filesystem;
	for (const auto& entry : fs::recursive_directory_iterator(root_dir)) {
		if (entry.is_regular_file()) {
			std::string filename = entry.path().filename().string();

			std::string feature_ext = ".feature";
			std::string pth_ext = ".pth";

			if ((filename.length() >= feature_ext.length() &&
				filename.compare(filename.length() - feature_ext.length(), feature_ext.length(), feature_ext) == 0) ||
				(filename.length() >= pth_ext.length() &&
					filename.compare(filename.length() - pth_ext.length(), pth_ext.length(), pth_ext) == 0)) {
				std::string full_path = fs::absolute(entry.path()).string();

				outfile << full_path << " " << speaker_id << std::endl;
			}
		}
	}

	outfile.close();
	std::cout << "Generated path mapping file: " << output_file << std::endl;
	std::cout << "Using speaker ID: " << speaker_id << std::endl;
}