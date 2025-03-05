#pragma once
#include <string>
#include <vector>

class Filesearch
{
public:
	//返回某路径下所有wav文件的路径
	static std::vector<std::string> getAllFiles(const std::string& url);
};

