#include "constants.h"

#include <vector>

#include "HandlerSingleton.h"
#include "Tensor.h"
#include "Layers/FullyConnectedLayer.h"
#include "Layers/SoftMaxLayer.h"
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
	SoftMaxLayer ac;
	CrossEntropyLoss ceLoss = CrossEntropyLoss(false);
	
	auto fc_out = fc1.forward(a);
	auto softmaxOut = ac.forward(fc_out);
	fc_out->ToHost();
	std::cout << "FC OUT: " << std::endl;
	printTensor(*fc_out);
	std::cout << "Softmax: " << std::endl;
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
	printTensor(*gradSoft);
	std::shared_ptr<Tensor> fcGrad = fc1.backward(gradSoft);

	//std::cout << "fc1 weight grad: " << std::endl;
	//std::cout << "fc1 Bias grad: " << std::endl;
	//delete& fc1;
	//std::cout << "kek";
}


