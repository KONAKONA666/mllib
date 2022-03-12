#pragma once
#include "cuda_headers.cuh"
#include "Layer.h"

class CrossEntropyLoss: public Layer
{
	std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& in) override;
	std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& dOut) override;

};

