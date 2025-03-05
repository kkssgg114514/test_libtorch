#include "Filesearch.h"
#include <filesystem>
#include <iostream>

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
