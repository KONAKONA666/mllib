#include "HandlerSingleton.h"

std::unique_ptr<HandlerSingleton> HandlerSingleton::instance = nullptr;
cublasHandle_t HandlerSingleton::cublasHandler;
cudnnHandle_t HandlerSingleton::cudnnHandler;
curandGenerator_t HandlerSingleton::curandHandler;


HandlerSingleton::HandlerSingleton() {
	cublasCreate(&cublasHandler);
	cudnnCreate(&cudnnHandler);
	curandCreateGenerator(&curandHandler, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandHandler, constants::SEED);
}


HandlerSingleton::~HandlerSingleton() {}

HandlerSingleton& HandlerSingleton::getInstance() {
	if (instance == nullptr) {
		instance.reset(new HandlerSingleton());
	}
	return *instance;
}