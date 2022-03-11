#include "HandlerSingleton.h"

std::unique_ptr<HandlerSingleton> HandlerSingleton::instance = nullptr;
cublasHandle_t HandlerSingleton::cublasHandler;
cudnnHandle_t HandlerSingleton::cudnnHandler;


HandlerSingleton::HandlerSingleton() {
	cublasCreate(&cublasHandler);
	cudnnCreate(&cudnnHandler);
}


HandlerSingleton::~HandlerSingleton() {
	std::cout << "DELETE HANDLER";
}

HandlerSingleton& HandlerSingleton::getInstance() {
	if (instance == nullptr) {
		instance.reset(new HandlerSingleton());
	}
	return *instance;
}