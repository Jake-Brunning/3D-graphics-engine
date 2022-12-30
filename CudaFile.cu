#include "cuda_runtime.h";
#include "device_launch_parameters.h"
#include <iostream>;

#include "Vector.h"
#include "List.h"
#include "LinkFile.cuh"

__device__ double fovX; //The angle of the camera to the rightmost view
__device__ double fovY; //The angle of the camera to the upmost view

//moving the camera
__global__ void MoveVectors(Vector* d_vectors, double changeInXYZ, char axis, int N) 
{
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (i < N) {
		switch (axis) {
		case 'x':
			d_vectors[i].x += changeInXYZ;
			break;

		case 'y':
			d_vectors[i].y += changeInXYZ;
			break;

		case 'z':
			d_vectors[i].z += changeInXYZ;
			break;
		}
	}
}

__host__ Vector* setUpMoveVectors(double changeInXYZ, char axis, Vector* vectors, int N) {

	Vector* d_vectors;
	cudaMalloc(&d_vectors, sizeof(Vector) * N);
	cudaMemcpy(d_vectors, vectors, sizeof(Vector) * N, cudaMemcpyHostToDevice);

	const int numberOfThreads = 64;
	const int numberOfBlocks = (N / numberOfThreads) + 1;
	
	MoveVectors << <numberOfBlocks, numberOfThreads >> > (d_vectors, changeInXYZ, axis, N);

	Vector* h_vectors = new Vector[N];
	cudaMemcpy(h_vectors, d_vectors, sizeof(Vector) * N, cudaMemcpyDeviceToHost);

	cudaFree(d_vectors);
	return h_vectors;
}

__global__ void calculateFovValues(double fovInput, double yPixels, double xPixels) {
	fovX = fovInput;
	fovY = fovInput * (yPixels / xPixels);

	fovX = fovX / 2;
	fovY = fovY / 2;
}

__host__ void setUpFovValuesForGPU(double fovInput, double yPixels, double xPixels) {
	calculateFovValues << <1, 1 >> > (fovInput, yPixels, xPixels);
}


template <typename type>
__global__ void matrixMultiply(type* matrix1, type* matrix2, type* returnMatrix, int width, int height) {
	int row = (blockDim.x * blockIdx.x) + threadIdx.x;
	int col = (blockDim.y * blockIdx.y) + threadIdx.y;

	if (row < width && col < height) {
		int tempSum = 0;
		for (int x = 0; x < height; x++) {
			tempSum += matrix1[x * width + col] * matrix2[row * width + x];
		}
		returnMatrix[row * width + col] = tempSum;
	}
}



__global__ void rotateAndProject(Vector* d_vectors, double* d_rotationMatrix, int N, const int width, const int height) {
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (i < N) {
		double coordinates[3] = { d_vectors[i].x, d_vectors[i].y, d_vectors[i].z };
		double output[9];

		const dim3 blocks(1, 1);
		const dim3 threads(width, height);

		matrixMultiply<double> << <blocks, threads >> > (coordinates, d_rotationMatrix, output, width, height);
		
	}
}

__host__ Vector* setUpRotationAndProjection(double xRotation[9], double yRotation[9], double zRotation[9], Vector* vectors, int N) {
	

	//multiply matrixes together
	const int width = 3; //width of a rotation matrix
	const int height = 3; //height of a rotation matrix
	
	const dim3 threads(width, height); //sets up threads to be 2d in order to execute a 2d gpu call
	const dim3 blocks(1, 1); //only have 9 elements in array no matter what, so blocks can just be 1


	//multiply first 2 matrixes
	double* d_xRotationTimesYRotation;
	cudaMalloc(&d_xRotationTimesYRotation, sizeof(double) * width * height);
	matrixMultiply<double> << <blocks, threads >> > (xRotation, yRotation, d_xRotationTimesYRotation ,width, height );
	
	//multiply remaining matrixes
	double* d_rotationMatrix;
	cudaMalloc(&d_rotationMatrix, sizeof(double) * width * height);
	matrixMultiply<double> <<<blocks, threads>>> (d_xRotationTimesYRotation, zRotation, d_rotationMatrix, width, height);

	cudaFree(d_xRotationTimesYRotation);


	//rotate and project vectors set up
	Vector* d_vectors;
	cudaMalloc(&d_vectors, sizeof(Vector) * N);
	cudaMemcpy(d_vectors, vectors, sizeof(Vector) * N, cudaMemcpyHostToDevice);

	const int numberOfThreads = 64;
	const int numberOfBlocks = (N / numberOfThreads) + 1;

	rotateAndProject << <numberOfBlocks, numberOfThreads >> > (d_vectors, d_rotationMatrix, N, width, height);

	Vector* h_projectedVectors = new Vector[N];
	cudaMemcpy(h_projectedVectors, d_vectors, sizeof(Vector) * N, cudaMemcpyDeviceToHost);
	return vectors;
}



