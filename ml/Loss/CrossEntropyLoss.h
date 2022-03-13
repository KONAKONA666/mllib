#pragma once
#include "../cuda_headers.cuh"
#include "../Layers/Layer.h"
class CrossEntropyLoss
{
private:
	const HandlerSingleton* handlers;
	std::shared_ptr<Tensor> inTensor;
	std::shared_ptr<Tensor> y_gt;
	bool reduction_mean;
public:
	CrossEntropyLoss(bool);
	float forward(const std::shared_ptr<Tensor> in, const std::shared_ptr<Tensor> y);
	std::shared_ptr<Tensor> backward();
};

