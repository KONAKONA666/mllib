#include "constants.h"

#include <vector>

#include "HandlerSingleton.h"
#include "Tensor.h"
#include "Layers/FullyConnectedLayer.h"
#include "Layers/SoftMaxLayer.h"
#include "Layers/ReluLayer.h"
#include "Loss/CrossEntropyLoss.h"

int main() {
	std::shared_ptr<Tensor> a = std::make_shared<Tensor>(2, 3);

	a->data[0] = 0.5;
	a->data[1] = 0.2;
	a->data[2] = 0.1;
	a->data[3] = 1;
	a->data[4] = 2;
	a->data[5] = 1.5;


	a->ToDevice();
	const float al = 1;
	const float bt = 0;
	const float* alpha = &al;
	const float* beta = &bt;

	std::shared_ptr<Tensor> l = std::make_shared<Tensor>( 2, 2 );
	l->data[0] = 1.0f;
	l->data[1] = 0.0f;
	l->data[2] = 0.0f;
	l->data[3] = 1.0f;
	
	l->ToDevice();
	std::cout << "y_gt: " << std::endl;	
	printTensor(*l);
	std::cout << "X:" << std::endl;
	printTensor(*a);


	FullyConnectedLayer fc1(3, 2);
	FullyConnectedLayer fc2(2, 2);
	SoftMaxLayer ac;
	CrossEntropyLoss ceLoss = CrossEntropyLoss(false);
	ReluLayer relu = ReluLayer();

	auto fc_out = fc1.forward(a);
	auto relu_out = relu.forward(fc_out);
	fc_out->ToHost();
	std::cout << "FC OUT: " << std::endl;
	printTensor(*fc_out);
	std::cout << "Relu OUT: " << std::endl;
	relu_out->ToHost();
	printTensor(*relu_out);
	auto fc2_out = fc2.forward(relu_out);
	fc2_out->ToHost();
	std::cout << "FC OUT: " << std::endl;
	printTensor(*fc2_out);
	std::cout << "Softmax: " << std::endl;
	auto softmaxOut = ac.forward(fc2_out);
	softmaxOut->ToHost();
	printTensor(*softmaxOut);
	float loss = ceLoss.forward(softmaxOut, l);
	std::cout <<"LOSS: " << loss << std::endl;
	std::cout << "BACKPROP: " << std::endl;
	std::shared_ptr<Tensor> gradL = ceLoss.backward();
	std::cout << "dL\n";
	gradL->ToHost();
	printTensor(*gradL);
	std::shared_ptr<Tensor> gradSoft = ac.backward(gradL);
	gradSoft->ToHost();
	std::cout << "dSoft\n";
	printTensor(*gradSoft);
	auto fc2Grad = fc2.backward(gradSoft);
	std::cout << "dfc2\n";
	fc2Grad->ToHost();
	printTensor(*fc2Grad);
	auto reluGrad = relu.backward(fc2Grad);
	reluGrad->ToHost();
	std::cout << "dRelu\n";
	printTensor(*reluGrad);
	auto t = fc1.backward(reluGrad);
	//std::cout << "fc1 weight grad: " << std::endl;
	//std::cout << "fc1 Bias grad: " << std::endl;
	//delete& fc1;
	//std::cout << "kek";
	std::cout << "FC1:\n";
	fc1.print();
	std::cout << "FC2:\n";
	fc2.print();
}


