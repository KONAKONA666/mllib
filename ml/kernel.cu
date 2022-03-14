#include "constants.h"

#include <vector>

#include "HandlerSingleton.h"
#include "Tensor.h"
#include "Layers/FullyConnectedLayer.h"
#include "Layers/SoftMaxLayer.h"
#include "Layers/ReluLayer.h"
#include "Loss/CrossEntropyLoss.h"
#include "Layers/Conv2d.h"

int main() {
	std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::initializer_list<int>{1, 1, 4, 4});


	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			a->data[4 * i + j] = 4 * i + j;
		}
	}


	a->ToDevice();
	const float al = 1;
	const float bt = 0;
	const float* alpha = &al;
	const float* beta = &bt;

	std::shared_ptr<Tensor> l = std::make_shared<Tensor>(1, 2);
	l->data[0] = 0.0f;
	l->data[1] = 1.0f;



	l->ToDevice();
	std::cout << "y_gt: " << std::endl;
	//printTensor(*l);
	std::cout << "X:" << std::endl;
	printTensor(*a);

	Conv2d conv(1, 2, 2);
	FullyConnectedLayer fc1(18, 4), fc2(4, 2);
	ReluLayer relu;
	SoftMaxLayer softmax;
	CrossEntropyLoss ceLoss(false);
	
	auto x = conv.forward(a);
	x->ToHost();
	x->squeeze();
	x = fc1.forward(x);
	x = relu.forward(x);
	x = fc2.forward(x);
	x = softmax.forward(x);
	float loss = ceLoss.forward(x, l);
	std::cout << "LOSS: " << loss << std::endl;
	auto dx = ceLoss.backward();
	dx = softmax.backward(dx);
	dx = fc2.backward(dx);
	dx = relu.backward(dx);
	dx = fc1.backward(dx);
	printShape(*dx);
	dx = conv.backward(dx);
	system("pause");
}


