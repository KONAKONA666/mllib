#pragma once
#include <memory>
#include "../Tensor.h"
#include "../HandlerSingleton.h"

struct Param
{
	std::unique_ptr<Tensor> data;
	std::unique_ptr<Tensor> grad;
	int rows;
	int cols;
	Param(int n, int m) : rows(n), cols(m), data(new Tensor{ n, m }), grad(new Tensor{ n, m }) {};
};


struct FilterParam {
	std::unique_ptr<Tensor> data;
	std::unique_ptr<Tensor> grad;
	int inChannels;
	int outChannels;
	int kernelSize;
	cudnnFilterDescriptor_t desc;
	FilterParam(int inC, int outC, int kernel) : inChannels(inC), outChannels(outC), kernelSize(kernel) {
		cudnnCreateFilterDescriptor(&desc);
		cudnnSetFilter4dDescriptor(desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outChannels, inChannels, kernelSize, kernelSize);
		data = std::make_unique<Tensor>(std::initializer_list<int>{inC, outC, kernel, kernel});
	};
	~FilterParam() {
		cudnnDestroyFilterDescriptor(desc);
	}
};


class Layer
{
protected:
	using T = std::shared_ptr<Tensor>;
	const HandlerSingleton* handlers;
	T inTensor;
	T outTensor;
public:
	Layer() : handlers(&HandlerSingleton::getInstance()) {};
	//virtual ~Layer() {};
	virtual T forward(const T) = 0;
	virtual T backward(const T&) = 0;
};