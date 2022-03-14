#pragma once
#include "Layer.h"

class MaxPoolLayer : public Layer {
private:
	cudnnPoolingDescriptor_t desc;
	int size, stride;
public:
	MaxPoolLayer(int size, int stride, int padding);
	std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> in);
	std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& dOut);
};
