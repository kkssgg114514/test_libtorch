#include "dll.h"
//dll.cpp
#include "export.h"
extern "C" _declspec(dllexport)void train_model(std::string wavDir)
{
	trainModel2(wavDir);
}

extern "C" _declspec(dllexport)int test_model(std::string wavPath)
{
	return testModel(wavPath);
}
