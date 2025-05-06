#pragma once
#include <string>

//输入音频文件夹+数字编号,生成模型
void trainModel(std::string wavDir, int index);

void trainModel2(std::string wavDir);

//输入单个音频文件,判断是某个编号下的说话人
int testModel(std::string wavPath);
