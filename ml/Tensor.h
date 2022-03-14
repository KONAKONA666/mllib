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
	Tensor(const std::vector<int>&);
	Tensor(std::initializer_list<int> s);
	Tensor(int n, int m): Tensor{ n, m } {};
	Tensor(const Tensor& t);
	~Tensor();
	void allocateMemory();
	void ToDevice();
	void ToHost();
	void random_init(curandGenerator_t& rng);
	void fillOnes();
	void fillZeros();
	void squeeze();
	std::shared_ptr<Tensor> create_like() const;
	
	//Tensor* matrix_general_mul(const Tensor& B, const float* alpha, const float* beta, const cublasHandle_t& handle);
	template<class PTR>
	PTR matrix_mul(const Tensor& B, const cublasHandle_t& handle, bool transposeA = false, bool transposeB = false);
	//Tensor* matrix_add(const Tensor& B, const cublasHandle_t& handle);

};


void CUBLAS_mat_mul(float* A, float* B, float* AB, int n, int m, int k, const cublasHandle_t& handle, bool transposeA, bool transposeB);
template<class PTR>
PTR tensor_mat_mul(const Tensor& A, const Tensor& B, const cublasHandle_t& handle, bool transposeA, bool transposeB);
void printTensor(Tensor& t);
void printTensor2d( Tensor& t);
void printShape(const Tensor& t);