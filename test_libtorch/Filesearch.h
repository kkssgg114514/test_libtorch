#pragma once
#include <string>
#include <vector>

class Filesearch
{
public:
	//����ĳ·��������wav�ļ���·��
	static std::vector<std::string> getAllFiles(const std::string& url);

	static void generate_feature_paths(const std::string& root_dir, const std::string& output_file, int speaker_id);
};

