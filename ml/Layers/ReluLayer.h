#pragma once
#include "Layer.h"

class ReluLayer: public Layer
{
	cudnnActivationDescriptor_t desc;
public:
	ReluLayer();
	~ReluLayer();
	std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> in) override;
	std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& dOut) override;
};

