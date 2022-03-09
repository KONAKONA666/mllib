﻿#include "constants.h"

#include <vector>


#include "Tensor.h"
#include "FullyConnectedLayer.h"
#include "losskernels.cuh"

using namespace constants;


float calculate_cross_entropy(const Tensor& preds, const Tensor& y) {
	Tensor loss{ preds.shape[0], preds.shape[1] };
	loss.fillZeros();
	GPU_crossentropy_kernel(preds.d_data, y.d_data, loss.d_data, LABELS, BATCH_SIZE);
	loss.ToHost();
	float ans = 0;
	for (int i = 0; i < BATCH_SIZE; i++) {
		ans += loss.data[i];
	}
	return ans;
}

Tensor* calculate_cross_backwad(const Tensor& y_pred, const Tensor& y_gt) {
	Tensor* tmp = new Tensor{BATCH_SIZE, LABELS};
	GPU_crossentropy_back_kernel(y_pred.d_data, y_gt.d_data, tmp->d_data, LABELS, BATCH_SIZE);
	return tmp;
}


Tensor* softmaxForward(const Tensor& in, cudnnHandle_t& handle) {
	const float alpha = 1.0f;
	const float beta = 0.0;
	Tensor* tmp = new Tensor{BATCH_SIZE, LABELS};
	checkCUDNN(
		cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, in.desc, in.d_data, &beta, tmp->desc, tmp->d_data)
	);
	return tmp;
}


Tensor* softMaxBack(const Tensor& lossGrad, const Tensor& y, cudnnHandle_t& handle) {
	const float alpha = 1.0f;
	const float beta = 0.0;
	Tensor* tmp = new Tensor{ BATCH_SIZE, LABELS };
	checkCUDNN(
		cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, y.desc, y.d_data, lossGrad.desc, lossGrad.d_data, &beta, tmp->desc, tmp->d_data)
	);
	return tmp;
}


int main() {
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudnnHandle_t cudnnHandle;
	cudnnCreate(&cudnnHandle);
	Tensor a({ 2, 3 });
	Tensor b({ 2, 3 });
	Tensor id({3, 3});
	Tensor grad({ 2, 3 });

	a.data[0] = 0.5;
	a.data[1] = 0.2;
	a.data[2] = 0.1;
	a.data[3] = 1;
	a.data[4] = 2;
	a.data[5] = 1.5;


	a.ToDevice();
	b.ToDevice();
	id.ToDevice();
	const float al = 1;
	const float bt = 0;
	const float* alpha = &al;
	const float* beta = &bt;

	Tensor l{ 2, 2 };
	l.data[0] = 1.0f;
	l.data[1] = 0.0f;
	l.data[2] = 0.0f;
	l.data[3] = 1.0f;
	
	l.ToDevice();
	std::cout << "y_gt: " << std::endl;	
	printTensor(l);
	std::cout << "X:" << std::endl;
	printTensor(a);



	FullyConnectedLayer<BATCH_SIZE> fc1(3, 2, &handle);
	Tensor* fc_out = fc1.forward(a);
	fc_out->ToHost();
	std::cout << "FC OUT: " << std::endl;
	printTensor(*fc_out);
	Tensor* softmaxOut = softmaxForward(*fc_out, cudnnHandle);
	std::cout << "Softmax: " << std::endl;
	softmaxOut->ToHost();
	printTensor(*softmaxOut);
	float loss = calculate_cross_entropy(*softmaxOut, l);
	std::cout <<"LOSS: " << loss << std::endl;
	std::cout << "BACKPROP: " << std::endl;
	Tensor gradL = *calculate_cross_backwad(*softmaxOut, l);
	Tensor gradSoft = *softMaxBack(gradL, *softmaxOut, cudnnHandle);
	gradSoft.ToHost();
	printTensor(gradSoft);
	Tensor* fcGrad = fc1.backward(gradSoft);
	std::cout << "fc1 weight grad: " << std::endl;
	std::cout << "fc1 Bias grad: " << std::endl;
	system("pause");
}


