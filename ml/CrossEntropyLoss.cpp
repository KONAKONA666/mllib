#include "CrossEntropyLoss.h"
#include "losskernels.cuh"

float CrossEntropyLoss::forward(const std::shared_ptr<Tensor> in, const std::shared_ptr<Tensor> y) {
	inTensor = in;
	y_gt = y;
	float loss = 0;
	int m = y->shape[0];
	int n = y->shape[1];
	std::unique_ptr<Tensor> tmp = std::make_unique<Tensor>(m, 1);
	GPU_crossentropy_kernel(in->d_data, y->d_data, tmp->d_data, n, m);
	tmp->ToHost();
	for (int i = 0; i < m; i++) {
		loss += tmp->data[i];
	}
	if (reduction == SUM) {
		return loss;
	}
	else {
		return loss / m;
	}
}


std::shared_ptr<Tensor> CrossEntropyLoss::backward() {
	int m = inTensor->shape[0];
	int n = inTensor->shape[1];
	std::shared_ptr<Tensor> dL = std::make_shared<Tensor>(m, n);
	GPU_crossentropy_back_kernel(inTensor->d_data, y_gt->d_data, dL->d_data, n, m);
	return dL;
}