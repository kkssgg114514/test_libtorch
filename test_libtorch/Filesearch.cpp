#include "Filesearch.h"
#include <filesystem>
#include <iostream>

std::vector<std::string> Filesearch::getAllFiles(const std::string& url)
{
    std::vector<std::string> wavFiles;


    try {
        // 检查目录是否存在
        if (!std::filesystem::exists(url)) {
            std::cerr << "目录不存在: " << url << std::endl;
            return wavFiles;
        }

        // 递归遍历目录中的所有文件和子目录
        for (const auto& entry : std::filesystem::recursive_directory_iterator(url)) {
            // 检查是否为普通文件
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                // 转换为小写进行比较
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

                if (extension == ".wav") {
                    // 获取完整的绝对路径
                    wavFiles.push_back(std::filesystem::absolute(entry.path()).string());
                }
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "文件系统错误: " << e.what() << std::endl;
    }

    return wavFiles;

}
