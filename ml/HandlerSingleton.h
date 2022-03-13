#pragma once
#include <memory>
#include "constants.h"
#include "cuda_headers.cuh"


class HandlerSingleton
{
private:
	static cublasHandle_t cublasHandler;
	static cudnnHandle_t cudnnHandler;
	static curandGenerator_t curandHandler;
	static std::unique_ptr<HandlerSingleton> instance;
	HandlerSingleton();
public:
	cublasHandle_t* getCublasHandle() const {
		return &cublasHandler;
	};
	cudnnHandle_t* getCudnnHandle() const {
		return &cudnnHandler;
	};
	curandGenerator_t* getCurandGenerator() const {
		return &curandHandler;
	};
	static HandlerSingleton& getInstance();
	~HandlerSingleton();
};
