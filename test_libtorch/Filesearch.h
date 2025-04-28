#pragma once
#include <string>
#include <vector>

class Filesearch
{
public:
	//返回某路径下所有wav文件的路径
	static std::vector<std::string> getAllFiles(const std::string& url);

	static void generate_feature_paths(const std::string& root_dir, const std::string& output_file, int speaker_id);
};

