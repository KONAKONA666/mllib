#include "../cuda_headers.cuh"

//// (bs, l) -> R

__global__ void crossentropy_kernel(float* y_pred, float* y_gt, float* loss);
//
//
//// dcross/dp_1 =  
__global__ void crossentropy_back_kernel(float* y_pred, float* y_gt, float* grad);


void GPU_crossentropy_back_kernel(float* y_pred, float* y_gt, float* grad, int labels, int batch_size);

void GPU_crossentropy_kernel(float* y_pred, float* y_gt, float* loss, int labels, int batch_size);