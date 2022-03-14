#pragma once
#include "../constants.h"
#include "../Tensor.h"
#include "Layer.h"




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