#include "FullyConnectedLayer.h"

using namespace constants;



void FullyConnectedLayer<BATCH_SIZE>::print() {
	//ones.get().ToHost();
	bias.data->ToHost();
	weights.data->ToHost();
	std::cout << "W: " << std::endl;
	printTensor(*weights.data);
	std::cout << "B: " << std::endl;
	printTensor(*bias.data);
};


std::shared_ptr<Tensor> FullyConnectedLayer<BATCH_SIZE>::backward(const std::shared_ptr<Tensor>& dOut) {
	float alpha = 1;
	float beta = 0;
	cublasHandle_t* handle = handlers->getCublasHandle();
	weights.grad = inTensor->matrix_mul(*dOut, *handle, true, false);
	std::shared_ptr<Tensor> tmp = std::make_shared<Tensor>(*dOut->matrix_mul(*weights.data, *handle, false, true));
	return tmp;
}


std::shared_ptr<Tensor> FullyConnectedLayer<BATCH_SIZE>::forward(const std::shared_ptr<Tensor>& in)  {
	float alpha = 1.0f;
	float beta = 0.0f;
	inTensor = in;
	cublasHandle_t* handle = handlers->getCublasHandle();
	outTensor = std::make_shared<Tensor>(*in->matrix_mul(*weights.data, *handle, false, false));
	checkCudaErrors(cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, batch_size, outputDim, 1, &alpha, ones->d_data, batch_size, bias.data->d_data, 1, &alpha, outTensor->d_data, batch_size));
	return outTensor;
};



FullyConnectedLayer<BATCH_SIZE>::FullyConnectedLayer(int inDim, int outDim)
	:	batch_size(BATCH_SIZE), 
		ones(new Tensor{batch_size, 1}),
		inputDim(inDim), outputDim(outDim), 
		weights(Param(inDim, outDim)),
		bias(Param(outDim, 1))
{
	ones->fillOnes();
	weights.data->fillOnes();
	bias.data->fillOnes();
}
