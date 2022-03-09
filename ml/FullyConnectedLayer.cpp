#include "FullyConnectedLayer.h"

using namespace constants;



void FullyConnectedLayer<BATCH_SIZE>::print() {
	ones.ToHost();
	bias.data->ToHost();
	weights.data->ToHost();
	std::cout << "W: " << std::endl;
	printTensor(*weights.data);
	std::cout << "B: " << std::endl;
	printTensor(*bias.data);
};


Tensor* FullyConnectedLayer<BATCH_SIZE>::backward(Tensor& d_gradOut) {
	float alpha = 1;
	float beta = 0;
	weights.grad = _d_in->matrix_mul(d_gradOut, *handle, true, false);
	Tensor* tmp = d_gradOut.matrix_mul(*weights.data, *handle, false, true);
	return tmp;
}


Tensor* FullyConnectedLayer<BATCH_SIZE>::forward(Tensor& d_in) {
	float alpha = 1.0f;
	float beta = 0.0f;
	_d_in = (Tensor*)&d_in;
	Tensor* _d_out = d_in.matrix_mul(*weights.data, *handle, false, false);
	checkCudaErrors(cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, batch_size, outputDim, 1, &alpha, ones.d_data, batch_size, bias.data->d_data, 1, &alpha, _d_out->d_data, batch_size));
	return _d_out;
};


FullyConnectedLayer<BATCH_SIZE>::FullyConnectedLayer(int inDim, int outDim, cublasHandle_t* h)
	: batch_size(BATCH_SIZE), inputDim(inDim), outputDim(outDim), handle(h),
	d_out(Tensor{ batch_size, outDim }),
	weights(Param(inDim, outDim)),
	ones(Tensor{ batch_size , 1 }),
	bias(Param(1, outDim))
{
	ones.fillOnes();
	bias.data->fillOnes();
	weights.data->fillOnes();
	d_out.fillZeros();
	d_out.ToHost();
};