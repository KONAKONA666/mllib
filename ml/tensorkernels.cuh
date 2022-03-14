#include "cuda_headers.cuh"

__global__ void fill_ones(float* M, int size);

__global__ void fill_zeros(float* M, int size);

__global__ void add_tensors(float* A, float* B, float* C, int size);

void GPU_fillOnes(float* d_data, int size);

void GPU_fillZeros(float* d_data, int size);

void GPU_addTensors(float* A, float* B, float* C, int size);