#include "Tensor.h"


void CUBLAS_mat_mul(float* A, float* B, float* AB, int m, int n, int k, const cublasHandle_t& handle, bool transposeA = false, bool transposeB = false ) {
}

Tensor::Tensor(const std::vector<int>& s): shape(s) {
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
};


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
	cudaMemcpy(d_data, t.d_data, _memorySize, cudaMemcpyDeviceToDevice);
	for (int i = 0; i < _size; i++) {
		data[i] = t.data[i];
	}
	cudaMemcpy(d_data, t.d_data, _memorySize, cudaMemcpyDeviceToDevice);
};


Tensor::~Tensor() {
	delete[] data;
	cudaFree(d_data);
	cudnnDestroyTensorDescriptor(desc);
};


void Tensor::allocateMemory() {
	data = new float[_memorySize];
	checkCudaErrors(cudaMalloc(&d_data, _memorySize));
};


void Tensor::ToDevice() {
	checkCudaErrors(cudaMemcpy(d_data, data, _memorySize, cudaMemcpyHostToDevice));
};


void Tensor::ToHost() {
	checkCudaErrors(cudaMemcpy(data, d_data, _memorySize, cudaMemcpyDeviceToHost));
};

void Tensor::random_init(curandGenerator_t& rng) {
	curandGenerateNormal(rng, d_data, _size, 0.0f, 1.0f);
};


void Tensor::fillOnes() {
	GPU_fillOnes(d_data, _size);
};

void Tensor::fillZeros() {
	GPU_fillZeros(d_data, _size);
};


void Tensor::squeeze() {
	nDim = 2;
	shape[1] = shape[1] * shape[2] * shape[3];
	shape[2] = 1;
	shape[3] = 1;
	int n = shape[0];
	int c = shape[1];
	int h = shape[2];
	int w = shape[3];
	//checkCUDNN(cudnnDestroyTensorDescriptor(desc));
	//checkCUDNN(cudnnCreateTensorDescriptor(&desc));
	//checkCUDNN(
	//	cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w)
	//);
}


std::shared_ptr<Tensor> Tensor::create_like() const {
	return std::shared_ptr<Tensor>(new Tensor(shape));
}

template<>
std::unique_ptr<Tensor> Tensor::matrix_mul(const Tensor& B, const cublasHandle_t& handle, bool transposeA, bool transposeB) {
	return tensor_mat_mul<std::unique_ptr<Tensor>>(*this, B, handle, transposeA, transposeB);
}

template<>
std::shared_ptr<Tensor> Tensor::matrix_mul(const Tensor& B, const cublasHandle_t& handle, bool transposeA, bool transposeB) {
	return tensor_mat_mul<std::shared_ptr<Tensor>>(*this, B, handle, transposeA, transposeB);
}


//(4, 4). (4, 4). m = 4
// n = 2
// k = 4
template<class PTR>
PTR tensor_mat_mul(const Tensor& A, const Tensor& B, const cublasHandle_t& handle, bool transposeA = false, bool transposeB = false) {
	int n, m, k;
	
	m = B.shape[1];
	n = A.shape[0];
	k = B.shape[0];

	if (transposeA) {
		n = A.shape[1];
	}
	if (transposeB) {
		m = B.shape[0];
		k = B.shape[1];
	}

	PTR AB(new Tensor(n, m));
	
	cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

	const float alpha = 1.0f;
	const float beta = 0.0f;


	cublasSgemm(handle, opB, opA, m, n, k, &alpha, B.d_data, B.shape[1], A.d_data, A.shape[1], &beta, AB->d_data, m);
	
	return AB;
}


void printTensor2d(Tensor& t) {
	for (int i = 0; i < t.shape[0]; i++) {
		for (int j = 0; j < t.shape[1]; j++) {
			std::cout << t.data[i * t.shape[1] + j]<<" ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
void printTensor(Tensor& t) {
	for (int i = 0; i < t.shape[0]; i++) {
		for (int j = 0; j < t.shape[1]; j++) {
			for (int k = 0; k < t.shape[2]; k++) {
				for (int p = 0; p < t.shape[3]; p++) {
					
					std::cout << t.data[t.shape[1] * t.shape[2] * t.shape[3] * i + t.shape[2] * t.shape[3] * j + t.shape[3] * k + p] << " ";
				}
				std::cout<<std::endl;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}


void printShape(const Tensor& t) {
	for (int i = 0; i < t.nDim; i++) {
		std::cout << t.shape[i] << " ";
	}
	std::cout << std::endl;
}