#include "tensorkernels.cuh"


__global__ void fill_ones(float* M, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		M[idx] = 1.0f;
	}
}

__global__ void fill_zeros(float* M, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		M[idx] = 0.0f;
	}
}

__global__ void add_tensors(float* A, float* B, float* C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		C[idx] = A[idx] + B[idx];
	}
}


void GPU_fillOnes(float* d_data, int size) {
	fill_ones << <128, 128 >> > (d_data, size);
}


void GPU_fillZeros(float* d_data, int size) {
	fill_zeros << <128, 128 >> > (d_data, size);
}


void GPU_addTensors(float* A, float* B, float* C, int size) {
	add_tensors << <128, 128 >> > (A, B, C, size);
}