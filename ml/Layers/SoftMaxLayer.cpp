#include "../cuda_headers.cuh"
#include "SoftMaxLayer.h"


std::shared_ptr<Tensor> SoftMaxLayer::forward(const std::shared_ptr<Tensor> in) {
	const float alpha = 1.0f;
	const float beta = 0.0;
	int m = in->shape[0];
	int n = in->shape[1];
	if(outTensor == nullptr)
		outTensor = std::make_shared<Tensor>(m, n);
	cudnnHandle_t* handle = handlers->getCudnnHandle();
	checkCUDNN(
		cudnnSoftmaxForward(*handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, in->desc, in->d_data, &beta, outTensor->desc, outTensor->d_data)
	);
	return outTensor;
};


std::shared_ptr<Tensor> SoftMaxLayer::backward(const std::shared_ptr<Tensor>& dOut) {
	const float alpha = 1.0f;
	const float beta = 0.0;
	int m = outTensor->shape[0];
	int n = outTensor->shape[1];
	//std::cout << "SOFTMAX BACK" << std::endl;
	if(dNext == nullptr)
		dNext = std::make_shared<Tensor>(m, n);
	
	cudnnHandle_t* handle = handlers->getCudnnHandle();
	checkCUDNN(
		cudnnSoftmaxBackward(*handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, outTensor->desc, outTensor->d_data, dOut->desc, dOut->d_data, &beta, dNext->desc, dNext->d_data)
	);
	return dNext;
};


