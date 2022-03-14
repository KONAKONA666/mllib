#include "Conv2d.h"


void Conv2d::getWorkspaceSize(size_t& workspace) {
	cudnnHandle_t* handle = handlers->getCudnnHandle();
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(*handle, inTensor->desc, weights.desc, desc, outTensor->desc, CUDNN_CONVOLUTION_FWD_ALGO_FFT, &workspace));

}

void Conv2d::getOutputDims(const std::shared_ptr<Tensor>& x, int& n, int& c, int& h, int& w) {
	cudnnGetConvolution2dForwardOutputDim(
		desc, x->desc, weights.desc, &n, &c, &h, &w
	);
}


Conv2d::Conv2d(int inC, int outC, int kernel): weights(inC, outC, kernel), bias(1, outC) {
	cudnnCreateConvolutionDescriptor(&desc);
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
	weights.data->fillZeros();
	bias.data->fillZeros();
}


std::shared_ptr<Tensor> Conv2d::forward(const std::shared_ptr<Tensor> in) {
	float alpha = 1;
	float beta = 0;
	inTensor = in;
	cudnnHandle_t* handle = handlers->getCudnnHandle();
	int n, c, h, w;
	getOutputDims(inTensor, n, c, h, w);
	outTensor = std::make_shared<Tensor>(std::initializer_list<int>{ n, c, h, w });
	if (d_workspace == nullptr) {
		getWorkspaceSize(workspaceSize);
		cudaMalloc(&d_workspace, workspaceSize);
	}
	checkCUDNN(cudnnConvolutionForward(*handle, &alpha, in->desc, in->d_data, weights.desc, weights.data->d_data, desc, CUDNN_CONVOLUTION_FWD_ALGO_FFT, d_workspace, workspaceSize, &beta, outTensor->desc, outTensor->d_data));
	checkCUDNN(cudnnAddTensor(*handle, &alpha, bias.data->desc, bias.data->d_data, &alpha, outTensor->desc, outTensor->d_data));
	return outTensor;
}


std::shared_ptr<Tensor> Conv2d::backward(const std::shared_ptr<Tensor>& dOut) {
	float alpha = 1;
	float beta = 0;
	cudnnHandle_t* handle = handlers->getCudnnHandle();
	size_t bwdSize, bwdDataSize;
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(*handle, inTensor->desc, outTensor->desc, desc, weights.desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT, &bwdSize));
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(*handle, weights.desc, outTensor->desc, desc, inTensor->desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT, &bwdDataSize));

	if (bwdSize > workspaceSize || bwdDataSize > workspaceSize) {
		workspaceSize = std::max(bwdSize, bwdDataSize);
		cudaFree(d_workspace);
		cudaMalloc(&d_workspace, workspaceSize);
	}

	std::shared_ptr<Tensor> tmp = inTensor->create_like();
	
	checkCUDNN(
		cudnnConvolutionBackwardBias(*handle, &alpha, outTensor->desc, dOut->d_data, &beta, bias.grad->desc, bias.grad->d_data)
	);
	checkCUDNN(	
		cudnnConvolutionBackwardFilter(*handle, &alpha, inTensor->desc, inTensor->d_data, outTensor->desc, dOut->d_data, desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT, d_workspace, workspaceSize, &beta, weights.desc, weights.grad->d_data)
	);
	checkCUDNN(
		cudnnConvolutionBackwardData(*handle, &alpha, weights.desc, weights.data->d_data, outTensor->desc, dOut->d_data, desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT, d_workspace, workspaceSize, &beta, tmp->desc, tmp->d_data)
	);
	return tmp;
}


Conv2d::~Conv2d() {
	cudaFree(d_workspace);
	cudnnDestroyConvolutionDescriptor(desc);
}