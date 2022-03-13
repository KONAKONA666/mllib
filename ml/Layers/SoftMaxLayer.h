#pragma once
#include "../constants.h"
#include "Layer.h"

class SoftMaxLayer: public Layer
{
public:
	std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> in) override;
	std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& dOut) override;
};

