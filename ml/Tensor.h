#pragma once
#include "tensorkernels.cuh"
#include <vector>


class Tensor
{
private:
	int _size;
	int _memorySize;
public:
	std::vector<int> shape;
	std::vector<int> strides;
	cudnnTensorDescriptor_t desc;
	float* data;
	float* d_data;
	int nDim;
	Tensor() {};
	Tensor(std::initializer_list<int> s);
	
	~Tensor();
	void allocateMemory();
	void ToDevice();
	void ToHost();
	void random_init(curandGenerator_t& rng);
	void fillOnes();
	void fillZeros();
	
	//Tensor* matrix_general_mul(const Tensor& B, const float* alpha, const float* beta, const cublasHandle_t& handle);
	Tensor* matrix_mul(const Tensor& B, const cublasHandle_t& handle, bool transposeA, bool transposeB);
	//Tensor* matrix_add(const Tensor& B, const cublasHandle_t& handle);

};


void CUBLAS_mat_mul(float* A, float* B, float* AB, int n, int m, int k, const cublasHandle_t& handle, bool transposeA, bool transposeB);
Tensor* tensor_mat_mul(const Tensor& A, const Tensor& B, const cublasHandle_t& handle, bool transposeA, bool transposeB);
void printTensor(Tensor& t);