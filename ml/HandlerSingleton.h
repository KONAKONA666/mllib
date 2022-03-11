#pragma once
#include <memory>
#include "cuda_headers.cuh"


class HandlerSingleton
{
private:
	static cublasHandle_t cublasHandler;
	static cudnnHandle_t cudnnHandler;
	static std::unique_ptr<HandlerSingleton> instance;
	HandlerSingleton();
public:
	cublasHandle_t* getCublasHandle() const {
		return &cublasHandler;
	}
	cudnnHandle_t* getCudnnHandle() const {
		return &cudnnHandler;
	}
	static HandlerSingleton& getInstance();
	~HandlerSingleton();
};
