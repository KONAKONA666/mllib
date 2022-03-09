#include "losskernels.cuh"


__global__ void crossentropy_back_kernel(float* y_pred, float* y_gt, float* grad) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int batchId = blockIdx.x;
	int labelId = threadIdx.x;
	if (y_gt[tid]) {
		grad[tid] = -1.0 / y_pred[tid];
	}
	else {
		grad[tid] = 0.0f;
	}
}



__global__ void crossentropy_kernel(float* y_pred, float* y_gt, float* loss) {
	__shared__ float entropies[3];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int sharedIdx = threadIdx.x;
	int batchId = blockIdx.x;
	if (y_gt[tid] > 0.0f) {
		entropies[sharedIdx] = -logf(y_pred[tid]);
	}
	else {
		entropies[sharedIdx] = 0.0f;
	}
	__syncthreads();
	if (sharedIdx == 0) {
		float sum = 0.0f;
		for (int i = 0; i < 3; i++) {
			sum += entropies[i];
		}
		loss[batchId] = sum;
	}
}



void GPU_crossentropy_back_kernel(float* y_pred, float* y_gt, float* grad, int labels, int batch_size) {
	crossentropy_back_kernel << < batch_size, labels >> > (y_pred, y_gt, grad);
}


void GPU_crossentropy_kernel(float* y_pred, float* y_gt, float* loss, int labels, int batch_size) {
	crossentropy_kernel << <batch_size, labels >> > (y_pred, y_gt, loss);
}