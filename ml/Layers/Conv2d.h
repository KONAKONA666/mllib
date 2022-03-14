#pragma once
#include "Layer.h"


class Conv2d: public Layer
{
private:

	cudnnConvolutionDescriptor_t desc;

	size_t workspaceSizeForward;
	size_t workspaceSizeBackward;
	int padH, padW;
	int strideH, strideV;
	int dilationH, dilationV;
	int inChannels, outChannels;
	int kernelSize;

	std::unique_ptr<FilterParam> weights;
	std::unique_ptr<Param> bias;

	int forwardH, forwardW;

	void* d_workspace = nullptr;
	size_t workspaceSize;

	void getOutputDims(const std::shared_ptr<Tensor>& x, int& n, int& c, int& h, int& w);
	void getWorkspaceSize(size_t&);

public:
//	Conv2d(int inC, int outC, int kernel, int pH = 0, int pW = 0, int sH = 1, int sW = 1, int dH = 1, int dV = 1);
	Conv2d(int, int, int);
	std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> in) override;
	std::shared_ptr<Tensor> backward(const std::shared_ptr<Tensor>& dOut) override;
};
