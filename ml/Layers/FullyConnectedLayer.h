#pragma once
#include "../constants.h"
#include "../Tensor.h"
#include "Layer.h"

struct Param
{
	std::unique_ptr<Tensor> data;
	std::unique_ptr<Tensor> grad;
	int rows;
	int cols;
	Param(int n, int m) : rows(n), cols(m), data(new Tensor{ n, m }), grad(new Tensor{ n, m }) {};
};



class FullyConnectedLayer: public Layer {
private:
	const int batch_size = constants::BATCH_SIZE;
	Param weights;
	Param bias;
	const int inputDim, outputDim;
	const std::unique_ptr<Tensor> ones;
public:
	using T = std::shared_ptr<Tensor>;
	FullyConnectedLayer(int inDim, int outDim);
	T forward(const T in) override;
	T backward(const T& dOut) override;
	void print();
};