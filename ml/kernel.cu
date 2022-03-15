#include "constants.h"

#include <vector>

#include "HandlerSingleton.h"
#include "Tensor.h"
#include "Layers/FullyConnectedLayer.h"
#include "Layers/SoftMaxLayer.h"
#include "Layers/ReluLayer.h"
#include "Loss/CrossEntropyLoss.h"
#include "Layers/Conv2d.h"
#include "Layers/MaxPoolLayer.h"

#include "cifar_reader.hpp"

#include <chrono>


using namespace constants;

struct VGG {
	std::vector<Param*> parameters;

	Conv2d conv1, conv2, conv3;
	MaxPoolLayer maxpool1, maxpool2;
	ReluLayer relu1, relu2, relu3, relu4;
	FullyConnectedLayer fc1, fc2;
	SoftMaxLayer softmax;


	VGG(): 
		conv1(Conv2d(3, 32, 3)), 
		maxpool1(MaxPoolLayer(2, 2, 0)), 
		conv2(Conv2d(32, 64, 3)), conv3(Conv2d(64, 64, 3)),
		maxpool2(2, 2, 0), fc1(FullyConnectedLayer(64*4*4, 64)), fc2(FullyConnectedLayer(64, 10)) {

		parameters.push_back((Param*)&conv1.weights);
		parameters.push_back((Param*)&conv1.bias);
		parameters.push_back((Param*)&conv2.weights);
		parameters.push_back((Param*)&conv2.bias);
		parameters.push_back((Param*)&conv3.weights);
		parameters.push_back((Param*)&conv3.bias);
		parameters.push_back((Param*)&fc1.weights);
		parameters.push_back((Param*)&fc1.bias);
		parameters.push_back((Param*)&fc2.weights);
		parameters.push_back((Param*)&fc2.bias);
	}
	

	std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
		auto y = conv1.forward(x);
		y = relu1.forward(y);
		y = maxpool1.forward(y);
		y = conv2.forward(y);
		y = relu2.forward(y);
		y = maxpool2.forward(y);
		y = conv3.forward(y);
		y = relu3.forward(y);
		y->squeeze();
		y = fc1.forward(y);
		y = relu4.forward(y);
		y = fc2.forward(y);
		y = softmax.forward(y);
		return y;
	}

	void backward(const std::shared_ptr<Tensor>& dL) {
		auto grad = softmax.backward(dL);
		grad = fc2.backward(grad);
		grad = relu4.backward(grad);
		grad = fc1.backward(grad);
		grad = relu3.backward(grad);
		grad = conv3.backward(grad);
		grad = maxpool2.backward(grad);
		grad = relu2.backward(grad);
		grad = conv2.backward(grad);
		grad = maxpool1.backward(grad);
		grad = relu1.backward(grad);
		grad = conv1.backward(grad);
	}

	void update(float lambda) {
		for (auto& param : parameters) {
			param->update(lambda);
		}
	}
	void zeroGrad() {
		for (auto& param : parameters) {
			param->grad->fillZeros();
		}
	}
	
};


int main() {
	auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	CrossEntropyLoss loss(true);
	VGG model;
	
	for (int epoch = 0; epoch < 10; epoch++) {
		float epochLoss = 0.0f;
		auto begin = std::chrono::high_resolution_clock::now();
		for (int batchInd = 0; batchInd < TRAIN_SIZE / BATCH_SIZE; batchInd++) {
			model.zeroGrad();
			std::shared_ptr<Tensor> X = std::make_shared<Tensor>(std::initializer_list<int>{BATCH_SIZE, 3, 32, 32});
			std::shared_ptr<Tensor> Y = std::make_shared<Tensor>(BATCH_SIZE, LABELS);
			Y->fillZeros();
			Y->ToHost();
			for (int ind = 0; ind < BATCH_SIZE; ind++) {
				int imageInd = BATCH_SIZE * batchInd + ind;
				for (int i = 0; i < 3 * 32 * 32; i++) {
					X->data[3 * 32 * 32 * ind + i] = (float)dataset.training_images[imageInd][i]/255.0f;
				}
				int label = dataset.training_labels[imageInd];
				Y->data[LABELS*ind + label] = 1.0f;
			}
			X->ToDevice();
			Y->ToDevice();
			auto out = model.forward(X);
			float currLoss = loss.forward(out, Y);
			epochLoss += currLoss/(TRAIN_SIZE/BATCH_SIZE);
			auto dL = loss.backward();
		//std::cout << "LOSS: " << currLoss << std::endl;

			model.backward(dL);
			model.update(learningRate / BATCH_SIZE);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
		printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
		std::cout << "LOSS: " << epochLoss << std::endl;
	}
	
	
	return 0;
}


