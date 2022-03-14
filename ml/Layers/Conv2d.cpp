#include "Conv2d.h"


void Conv2d::getWorkspaceSize(size_t& workspace) {
	cudnnHandle_t* handle = handlers->getCudnnHandle();
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(*handle, inTensor->desc, weights->desc, desc, outTensor->desc, CUDNN_CONVOLUTION_FWD_ALGO_FFT, &workspace));

}

void Conv2d::getOutputDims(const std::shared_ptr<Tensor>& x, int& n, int& c, int& h, int& w) {
	cudnnGetConvolution2dForwardOutputDim(
		desc, x->desc, weights->desc, &n, &c, &h, &w
	);
}


Conv2d::Conv2d(int inC, int outC, int kernel) {
	cudnnCreateConvolutionDescriptor(&desc);
	weights = std::make_unique<FilterParam>(inC, outC, kernel);
	padH = 0;
	padW = 0;
	strideH = 1;
	strideV = 1;
	dilationH = 1;
	dilationV = 1;
	inChannels = inC;
	kernelSize = kernel;
	outChannels = outC;
	cudnnSetConvolution2dDescriptor(desc, padH, padW, strideH, strideV, dilationH, dilationV, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
	weights->data->random_init(*handlers->getCurandGenerator());
}


std::shared_ptr<Tensor> Conv2d::forward(const std::shared_ptr<Tensor> in) {
	float alpha = 1;
	float beta = 0;
	inTensor = in;
	cudnnHandle_t* handle = handlers->getCudnnHandle();
	int n, c, h, w;
	getOutputDims(inTensor, n, c, h, w);
	outTensor = std::make_shared<Tensor>(std::initializer_list<int>{ n, c, h, w });
	printf("n: %d, c: %d, h: %d, w: %d", n, c, h, w);
	if (d_workspace == nullptr) {
		getWorkspaceSize(workspaceSize);
		cudaMalloc(&d_workspace, workspaceSize);
	}
	checkCUDNN(cudnnConvolutionForward(*handle, &alpha, in->desc, in->d_data, weights->desc, weights->data->d_data, desc, CUDNN_CONVOLUTION_FWD_ALGO_FFT, d_workspace, workspaceSize, &beta, outTensor->desc, outTensor->d_data));
	return outTensor;
}


std::shared_ptr<Tensor> Conv2d::backward(const std::shared_ptr<Tensor>& dOut) {
	return std::shared_ptr<Tensor>();
}

