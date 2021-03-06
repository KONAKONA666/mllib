#pragma once
#include "../constants.h"
#include "../Tensor.h"
#include "Layer.h"




class FullyConnectedLayer: public Layer {
private:
	const int batch_size = constants::BATCH_SIZE;
	const int inputDim, outputDim;
	const std::unique_ptr<Tensor> ones;
public:
	Param weights;
	Param bias;
	FullyConnectedLayer(int inDim, int outDim);
	std::shared_ptr<Tensor> forward(const  std::shared_ptr<Tensor> in) override;
	std::shared_ptr<Tensor> backward(const  std::shared_ptr<Tensor>& dOut) override;
	void print();
};