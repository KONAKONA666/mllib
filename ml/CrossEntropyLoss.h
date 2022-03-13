#pragma once
#include "cuda_headers.cuh"
#include "Layer.h"
enum REDUCTION {SUM, AVERAGE};
class CrossEntropyLoss
{
private:
	const HandlerSingleton* handlers;
	std::shared_ptr<Tensor> inTensor;
	std::shared_ptr<Tensor> y_gt;
	REDUCTION reduction;
public:
	float forward(const std::shared_ptr<Tensor> in, const std::shared_ptr<Tensor> y);
	std::shared_ptr<Tensor> backward();
};

