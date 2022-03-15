#include "ReluLayer.h"

ReluLayer::ReluLayer() {
	cudnnCreateActivationDescriptor(&desc);
	cudnnSetActivationDescriptor(desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);
}

ReluLayer::~ReluLayer() {
	cudnnDestroyActivationDescriptor(desc);
}

std::shared_ptr<Tensor> ReluLayer::forward(const std::shared_ptr<Tensor> in) {
	inTensor = in;
	if (outTensor == nullptr)
		outTensor = in->create_like();
	float alpha = 1;
	float beta = 0;
	cudnnHandle_t* handle = handlers->getCudnnHandle();
	cudnnActivationForward(*handle, desc, &alpha, in->desc, in->d_data, &beta, outTensor->desc, outTensor->d_data);
	return outTensor;
}


std::shared_ptr<Tensor> ReluLayer::backward(const std::shared_ptr<Tensor>& dOut) {
	float alpha = 1;
	float beta = 0;
	cudnnHandle_t* handle = handlers->getCudnnHandle();
	if(dNext == nullptr)
		dNext = std::make_shared<Tensor>(inTensor->shape);
	cudnnActivationBackward(*handle, desc, &alpha, outTensor->desc, outTensor->d_data, dOut->desc, dOut->d_data, inTensor->desc, inTensor->d_data, &beta, dNext->desc, dNext->d_data);
	return dNext;
}
