#include "FullyConnectedLayer.h"

using namespace constants;



void FullyConnectedLayer::print() {
	//ones.get().ToHost();
	bias.data->ToHost();
	weights.data->ToHost();
	weights.grad->ToHost();
	std::cout << "W: " << std::endl;
	printTensor(*weights.data);
	std::cout << "B: " << std::endl;
	printTensor(*bias.data);
	std::cout << "dW: " << std::endl;
	printTensor(*weights.grad);
};


//[bs, out] x [out, in] => [bs, in]
std::shared_ptr<Tensor> FullyConnectedLayer::backward(const std::shared_ptr<Tensor>& dOut) {
	float alpha = 1;
	float beta = 0;
	cublasHandle_t* handle = handlers->getCublasHandle();
	weights.grad = inTensor->matrix_mul<std::unique_ptr<Tensor>>(*dOut, *handle, true, false);
	bias.grad = ones->matrix_mul<std::unique_ptr<Tensor>>(*dOut, *handle, true, false);
	std::shared_ptr<Tensor> dNext = dOut->matrix_mul<std::shared_ptr<Tensor>>(*weights.data, *handle, false, true);
	return dNext;
}


std::shared_ptr<Tensor> FullyConnectedLayer::forward(const std::shared_ptr<Tensor> in)  {
	float alpha = 1.0f;
	float beta = 0.0f;
	inTensor = in;
	cublasHandle_t* handle = handlers->getCublasHandle();
	outTensor = in->matrix_mul<std::shared_ptr<Tensor>>(*weights.data, *handle, false, false);
	checkCudaErrors(cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, batch_size, outputDim, 1, &alpha, ones->d_data, batch_size, bias.data->d_data, 1, &alpha, outTensor->d_data, batch_size));
	return outTensor;
}



FullyConnectedLayer::FullyConnectedLayer(int inDim, int outDim)
	:	batch_size(BATCH_SIZE), 
		ones(new Tensor{batch_size, 1}),
		inputDim(inDim), outputDim(outDim), 
		weights(Param(inDim, outDim)),
		bias(Param(outDim, 1))
{
	ones->fillOnes();
	weights.data->random_init(*handlers->getCurandGenerator());
	bias.data->random_init(*handlers->getCurandGenerator());
}
