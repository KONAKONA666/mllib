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
	std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::initializer_list<int>{ 1, 1, 4, 4 });

	
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

	std::shared_ptr<Tensor> l = std::make_shared<Tensor>( 2, 2 );
	l->data[0] = 1.0f;
	l->data[1] = 0.0f;
	l->data[2] = 0.0f;
	l->data[3] = 1.0f;
	
	l->ToDevice();
	std::cout << "y_gt: " << std::endl;	
	//printTensor(*l);
	std::cout << "X:" << std::endl;
	printTensor(*a);

	Conv2d conv(1, 2, 2);
	auto f = conv.forward(a);
	f->ToHost();
	std::cout << std::endl;
	printTensor(*f);
	system("pause");
}


