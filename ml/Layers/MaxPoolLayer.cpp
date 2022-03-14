#include "MaxPoolLayer.h"


MaxPoolLayer::MaxPoolLayer(int size, int stride, int padding = 0): size(size), stride(stride) {
	cudnnCreatePoolingDescriptor(&desc);
	cudnnSetPooling2dDescriptor(
		desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, size, size, padding, padding, stride, stride
	);
}


std::shared_ptr<Tensor> MaxPoolLayer::forward(const std::shared_ptr<Tensor> in) {
	inTensor = in;
	int n, c, h, w;
	float alpha = 1;
	float beta = 0;
	cudnnHandle_t* handle = handlers->getCudnnHandle();
	cudnnGetPooling2dForwardOutputDim(desc, inTensor->desc, &n, &c, &h, &w);
	outTensor = std::make_unique<Tensor>(std::initializer_list<int>{n, c, h, w});
	cudnnPoolingForward(
		*handle, desc, &alpha, inTensor->desc, inTensor->d_data, &beta, outTensor->desc, outTensor->d_data
	);
	return outTensor;
}


std::shared_ptr<Tensor> MaxPoolLayer::backward(const std::shared_ptr<Tensor>& dOut) {
	float alpha = 1;
	float beta = 0;
	cudnnHandle_t* handle = handlers->getCudnnHandle();
	std::shared_ptr<Tensor> tmp = inTensor->create_like();
	cudnnPoolingBackward(
		*handle, desc, &alpha, outTensor->desc, outTensor->d_data, dOut->desc, dOut->d_data, inTensor->desc, inTensor->d_data, &beta, tmp->desc, tmp->d_data
	);
	return tmp;
}