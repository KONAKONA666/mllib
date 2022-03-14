#pragma once
#include <memory>
#include "../Tensor.h"
#include "../HandlerSingleton.h"


struct Param
{
	const HandlerSingleton* handlers;
	std::unique_ptr<Tensor> data;
	std::unique_ptr<Tensor> grad;
	std::vector<int> shape;
	Param() { handlers = &HandlerSingleton::getInstance(); };
	Param(int n, int m) : data(new Tensor{ n, m }), grad(new Tensor{ n, m }) {
		shape = { n, m };
		handlers = &HandlerSingleton::getInstance();
	};
	Param(std::vector<int> s) : shape(s) {
		handlers = &HandlerSingleton::getInstance();
		data = std::make_unique<Tensor>(shape);
		grad = std::make_unique<Tensor>(shape);
	}
	void update(float lamda) {
		float alpha = -lamda;
		float beta = 1.0f;
		cublasHandle_t* handle = handlers->getCublasHandle();
		cublasSaxpy(
			*handle, data->getSize(), &alpha, grad->d_data, 1, data->d_data, 1
		);
	}
};


struct FilterParam: public Param {
	cudnnFilterDescriptor_t desc;
	FilterParam(int inC, int outC, int kernel) {
		shape = { inC, outC, kernel, kernel };
		cudnnCreateFilterDescriptor(&desc);
		cudnnSetFilter4dDescriptor(desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outC, inC, kernel, kernel);
		data = std::make_unique<Tensor>(std::initializer_list<int>{inC, outC, kernel, kernel});
		grad = std::make_unique<Tensor>(std::initializer_list<int>{inC, outC, kernel, kernel});
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
	virtual ~Layer() {};
	virtual T forward(const T) = 0;
	virtual T backward(const T&) = 0;
};