#pragma once
#include <string>

//������Ƶ�ļ���+���ֱ��,����ģ��
void trainModel(std::string wavDir, int index);

//���뵥����Ƶ�ļ�,�ж���ĳ������µ�˵����
int testModel(std::string wavPath);