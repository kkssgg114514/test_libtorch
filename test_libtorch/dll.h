//dll.h
#ifndef DLL_H_
#define DLL_H_
#include <string>
extern "C" _declspec(dllexport) void train_model(std::string wavDir);

extern "C" _declspec(dllexport) int test_model(std::string wavPath);
#endif
