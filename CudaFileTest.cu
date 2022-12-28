#include "cuda_runtime.h";
#include "device_launch_parameters.h"
#include "CudaLinkTest.cuh";
#include <iostream>;



__global__ void parrallelExecutionTest(int* d_A, int* d_B, int* d_C) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	d_C[i] = d_A[i] + d_B[i];
}

__host__ void cpuEnterFunction() {
	int N = 100;
	int* h_A = new int[N];
	int* h_B = new int[N];
	int* h_C = new int[N];

	for (int i = 0; i < N; i++) {
		h_A[i] = i;
		h_B[i] = i;
	}

	//initilises the cuda environment
	cudaFree(0);

	int* d_A;
	cudaMalloc(&d_A, sizeof(int) * N);
	int* d_B;
	cudaMalloc(&d_B, sizeof(int) * N);
	int* d_C;
	cudaMalloc(&d_C, sizeof(int) * N);

	cudaMemcpy(d_A, h_A, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeof(int) * N, cudaMemcpyHostToDevice);

	int amountOfBlocks = 4;
	parrallelExecutionTest << <amountOfBlocks, N / amountOfBlocks >> > (d_A, d_B, d_C);
	cudaMemcpy(h_C, d_C, sizeof(int) * N, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		std::cout << h_C[i] << std::endl;
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}