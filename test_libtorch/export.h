#pragma once
#include <string>

//������Ƶ�ļ���+���ֱ��,����ģ��
void trainModel(std::string wavDir, int index);

void trainModel2(std::string wavDir);

//���뵥����Ƶ�ļ�,�ж���ĳ������µ�˵����
int testModel(std::string wavPath);
