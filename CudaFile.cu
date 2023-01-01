#include "cuda_runtime.h";
#include "device_launch_parameters.h"
#include <iostream>;

#include "Vector.h"
#include "List.h"
#include "LinkFile.cuh"
#include "Camera.h"

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
	fovY = fovInput;
}

__host__ void setUpFovValuesForGPU(double fovInput, double yPixels, double xPixels) {
	calculateFovValues << <1, 1 >> > (fovInput, yPixels, xPixels);
}

template <typename type>
//this function assumes only valid matrixes are passed into it
__global__ void matrixMultiply(type* matrix1, type* matrix2, type* returnMatrix, const int matrix2Width, const int matrix2Height) {
	//each thread will handle one row and one column of the matrix multiplication
	

	int row = (blockDim.x * blockIdx.x) + threadIdx.x;
	int col = (blockDim.y * blockIdx.y) + threadIdx.y;

	if (row < matrix2Height && col < matrix2Width) {
		double tempSum = 0;
		for (int x = 0; x < matrix2Height; x++) {
			tempSum += matrix1[row * matrix2Height + x] * matrix2[x * matrix2Width + col];
		}
		returnMatrix[row * matrix2Width + col] = tempSum;
	}
}

template<typename type>
__device__ type modulus(type input) { //makes a negative positive
	if (input < 0) {
		return input * -1;
	}
	return input;
}

__device__ bool checkIfInViewFrustrum(Vector vec, double zDistFromNearClip) {
	if (zDistFromNearClip > vec.z) { //check if behind the near clip plane
		return false;
	}

	vec.z = modulus<double>(vec.z); //makes values for vectors positives
	vec.y = modulus<double>(vec.y);
	vec.x = modulus<double>(vec.x);
	
	double xAngle = atan(vec.x / vec.z); //Get angle which is formed with the axis
	double yAngle = atan(vec.y / vec.z);

	if (xAngle > fovX || yAngle > fovY) {
		return false;
	}

	return true;
}

__global__ void rotateAndProject(Vector* d_vectors, double* this_rotationMatrix, const int width, const int N, Camera camera) {
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (i < N) {
		//set up rotation call
		const int this_widthOfCoords = 1;
		const int this_heightOfCoords = 3;
		const int amountOfCoords = this_widthOfCoords * this_heightOfCoords;
		const dim3 blocks(1, 1);
		const dim3 threads(3, 3);

		double* d_rotated; //the rotated vectors
		double* d_coordsInMatrixForm; //the coordinates of a vector in matrix form

		cudaMalloc(&d_rotated, sizeof(double) * amountOfCoords);
		cudaMalloc(&d_coordsInMatrixForm, sizeof(double) * amountOfCoords);

		d_coordsInMatrixForm[0] = d_vectors[i].x;
		d_coordsInMatrixForm[1] = d_vectors[i].y;
		d_coordsInMatrixForm[2] = d_vectors[i].z;

		//perform rotation
		matrixMultiply<double> <<<blocks, threads >>> (this_rotationMatrix, d_coordsInMatrixForm, d_rotated, this_widthOfCoords, this_heightOfCoords);
		cudaDeviceSynchronize(); //stops code from continuing to execute until all vectors have been rotated

		//assign rotated x y z to vectors x y z
		d_vectors[i].x = d_rotated[0];
		d_vectors[i].y = d_rotated[1];
		d_vectors[i].z = d_rotated[2];

		//free memory space
		cudaFree(d_coordsInMatrixForm);
		cudaFree(d_rotated);

		//project vector if in view frustrum
		if (checkIfInViewFrustrum(d_vectors[i], camera.getDistanceZ())) {
			d_vectors[i].projectVector(camera.getDistanceX(), camera.getDistanceY(), camera.getDistanceZ());
		}

		__syncthreads(); //syncthreads needed so vectors dont get projected twice.

		short numOfVector = i % 3;
		if (d_vectors[i].getProjectVector() == true) {
			//project connected vectors

			short numOfVector = i % 3;

			//keep in mind each 3 consecutive vectors in d_vectors is a triangle
			//for loop will always start on a multiple of 3 and end before one. e.g:
			// 0,1,2 or 33,34,35 or 6,7,8.
			//so only connectd triangle to a projected vector is projected
			for (int x = i - numOfVector; x < i + 3; x++) {
				if (d_vectors[x].getProjectVector() == false) {
					d_vectors[x].projectVector(camera.getDistanceX(), camera.getDistanceY(), camera.getDistanceZ());
				}
			}
		}
		__syncthreads();
	}
}

__host__ Vector* setUpRotationAndProjection(double h_xRotation[9], double h_yRotation[9], double h_zRotation[9], Vector* h_vectors, const int N,  Camera camera) {

	//data for matrix multiply set up
	const int h_width = 3; //width of a rotation matrix
	const int h_height = 3; //height of a rotation matrix
	const size_t sizeOfMatrix = sizeof(double) * h_width * h_height; //the size in bytes of a matrix
	const dim3 threads(h_width, h_height); //2d thread setup
	const dim3 blocks(1, 1); //2d block setup. only need to use one as rotation matrix multiplication would only be as big as 9 threads (width * height)


	//multiply the first 2 matrixes
	double* d_XYoutput; //the x and y matrix multiplied together
	double* d_xRotation;
	double* d_yRotation; 
	cudaMalloc(&d_XYoutput, sizeOfMatrix);
	cudaMalloc(&d_xRotation, sizeOfMatrix);
	cudaMalloc(&d_yRotation, sizeOfMatrix);

	cudaMemcpy(d_xRotation, h_xRotation, sizeOfMatrix, cudaMemcpyHostToDevice);
	cudaMemcpy(d_yRotation, h_yRotation, sizeOfMatrix, cudaMemcpyHostToDevice);

	matrixMultiply<double> << <blocks, threads >> > (d_xRotation, d_yRotation, d_XYoutput, h_width, h_height);

	//multiply the XY result with the Z matrix;
	double* d_XYZoutput;
	double* d_zRotation;
	cudaMalloc(&d_XYZoutput, sizeOfMatrix);
	cudaMalloc(&d_zRotation, sizeOfMatrix);

	cudaMemcpy(d_zRotation, h_zRotation, sizeOfMatrix, cudaMemcpyHostToDevice);

	matrixMultiply<double> << <blocks, threads >> > (d_XYoutput, d_zRotation, d_XYZoutput, h_width, h_height);

	//testing to see if rotation matrix is the correct value
	double* h_XYZoutput = new double[h_width * h_height];
	cudaMemcpy(h_XYZoutput, d_XYZoutput, sizeOfMatrix, cudaMemcpyDeviceToHost);
	//end of test

	cudaFree(d_xRotation);
	cudaFree(d_yRotation);
	cudaFree(d_zRotation);
	cudaFree(d_XYoutput);

	//set up rotate vectors
	Vector* d_vectors;
	cudaMalloc(&d_vectors, sizeof(Vector) * N);

	cudaMemcpy(d_vectors, h_vectors, sizeof(Vector) * N, cudaMemcpyHostToDevice);

	const int numberOfThreads = 32; //arbituary value for number of threads, tbh this could be increased to 64,128 or 512 for faster processing
	const int numberOfBlocks = (N / numberOfThreads) + 1; //if more threads needed than can fit in a block, add another block

	rotateAndProject << <numberOfBlocks, numberOfThreads >> > (d_vectors, d_XYZoutput, h_width, N, camera);

	Vector* h_output = new Vector[N]; //the output of rotate and projecet (all vectors, except vectors in view of frustrum will be projected)
	cudaMemcpy(h_output, d_vectors, sizeof(Vector) * N, cudaMemcpyDeviceToHost);

	cudaFree(d_vectors);
	cudaFree(d_XYZoutput);

	return h_output;
}

