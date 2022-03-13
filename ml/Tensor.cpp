#include "Tensor.h"


void CUBLAS_mat_mul(float* A, float* B, float* AB, int n, int m, int k, const cublasHandle_t& handle, bool transposeA = false, bool transposeB = false) {
	cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

	int c_n = m;
	int c_m = n;
	int c_k = k;

	const float alpha = 1.0f;
	const float beta = 0.0f;

	if (transposeA) {
		c_n = m;
		c_m = k;
		c_k = n;
	}

	cublasSgemm(handle, opB, opA, c_n, c_m, c_k, &alpha, B, m, A, k, &beta, AB, m);
}


Tensor::Tensor(std::initializer_list<int> s) : shape(s) {
	nDim = (int)shape.size();
	std::vector<int> _shape = shape;
	while (_shape.size() < 4) _shape.push_back(1);
	int n = _shape[0];
	int c = _shape[1];
	int h = _shape[2];
	int w = _shape[3];
	_size = n * c * h * w;
	_memorySize = _size * sizeof(float);
	allocateMemory();
	checkCUDNN(cudnnCreateTensorDescriptor(&desc));
	checkCUDNN(
		cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)
	);
}


Tensor::Tensor(const Tensor& t) {
	nDim = t.nDim;
	shape = t.shape;
	std::vector<int> _shape = shape;
	while (_shape.size() < 4) _shape.push_back(1);
	int n = _shape[0];
	int c = _shape[1];
	int h = _shape[2];
	int w = _shape[3];
	_size = n * c * h * w;
	_memorySize = _size * sizeof(float);
	allocateMemory();
	checkCUDNN(cudnnCreateTensorDescriptor(&desc));
	checkCUDNN(
		cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)
	);
	for (int i = 0; i < _size; i++) {
		data[i] = t.data[i];
	}
	cudaMemcpy(d_data, t.d_data, _memorySize, cudaMemcpyDeviceToDevice);
}


Tensor::~Tensor() {
	delete[] data;
	cudaFree(d_data);
	cudnnDestroyTensorDescriptor(desc);
}


void Tensor::allocateMemory() {
	data = new float[_memorySize];
	checkCudaErrors(cudaMalloc(&d_data, _memorySize));
}


void Tensor::ToDevice() {
	checkCudaErrors(cudaMemcpy(d_data, data, _memorySize, cudaMemcpyHostToDevice));
}


void Tensor::ToHost() {
	checkCudaErrors(cudaMemcpy(data, d_data, _memorySize, cudaMemcpyDeviceToHost));
}

void Tensor::random_init(curandGenerator_t& rng) {
	curandGenerateUniform(rng, d_data, _size);
};


void Tensor::fillOnes() {
	GPU_fillOnes(d_data, _size);
};

void Tensor::fillZeros() {
	GPU_fillZeros(d_data, _size);
};

template<>
std::unique_ptr<Tensor> Tensor::matrix_mul(const Tensor& B, const cublasHandle_t& handle, bool transposeA, bool transposeB) {
	return tensor_mat_mul<std::unique_ptr<Tensor>>(*this, B, handle, transposeA, transposeB);
}

template<>
std::shared_ptr<Tensor> Tensor::matrix_mul(const Tensor& B, const cublasHandle_t& handle, bool transposeA, bool transposeB) {
	return tensor_mat_mul<std::shared_ptr<Tensor>>(*this, B, handle, transposeA, transposeB);
}

// (2, 3) x (2, 3). (3, 2) x (2, 3) -> (3, 3)
// (3, 2) x (3, 2). (2, 3) x (3, 2) -> (2, 2)
// (2, 3) x (3, 2)
template<class PTR>
PTR tensor_mat_mul(const Tensor& A, const Tensor& B, const cublasHandle_t& handle, bool transposeA = false, bool transposeB = false) {
	int n = A.shape[0];
	int m = B.shape[1];
	int k = A.shape[1];
	int dimN = n;
	int dimM = m;
	if (transposeA) {
		dimN = k;
	}
	if (transposeB) {
		dimM = k;
	}
	PTR AB(new Tensor(dimN, dimM));
	CUBLAS_mat_mul(A.d_data, B.d_data, AB->d_data, n, m, k, handle, transposeA, transposeB);
	return AB;
}


void printTensor(Tensor& t) {
	for (int i = 0; i < t.shape[0]; i++) {
		for (int j = 0; j < t.shape[1]; j++) {
			std::cout << t.data[i * t.shape[1] + j] << " ";
		}
		std::cout << std::endl;
	}
}
