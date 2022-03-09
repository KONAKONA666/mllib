#pragma once
#include "constants.h"
#include "Tensor.h"

struct Param
{
	Tensor* data;
	Tensor* grad;
	int rows;
	int cols;
	Param(int n, int m) : rows(n), cols(m) {
		data = new Tensor{ n, m };
		grad = new Tensor{ n, m };
	};
};

template<int batch_size>
class FullyConnectedLayer {
private:
	int batch_size = constants::BATCH_SIZE;
	Param weights;
	Param bias;
	Tensor d_out;
	Tensor ones;
	Tensor* _d_in = nullptr;
	int inputDim, outputDim;
	cublasHandle_t* handle;
public:
	// (out, batch_size) + 
	// (out, in)x(in, batch_size) -> (out, batch_size)
	FullyConnectedLayer(int inDim, int outDim, cublasHandle_t* h);
	Tensor* forward(Tensor& d_in);
	Tensor* backward(Tensor& d_gradOut);
	void print();
};