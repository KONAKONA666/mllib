#pragma once
#include <memory>
#include "Tensor.h"
#include "HandlerSingleton.h"


class Layer
{
protected:
	using T = std::shared_ptr<Tensor>;
	const HandlerSingleton* handlers;
	T inTensor;
	T outTensor;
public:
	Layer() : handlers(&HandlerSingleton::getInstance()) {};
	virtual T forward(const T&) = 0;
	virtual T backward(const T&) = 0;
};