#include "cuda_runtime.h";
#include "device_launch_parameters.h"
#include <iostream>;

#include "Vector.h"
#include "List.h"
#include "LinkFile.cuh"
#include "Camera.h"
#include "Pixel.h"

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

	const int numberOfThreads = 512;
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
		const dim3 threads(3, 1);

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
			for (int x = 0; x < 3; x++) {
				if (d_vectors[x + i - numOfVector].getProjectVector() == false) {
					d_vectors[x + i - numOfVector].projectVector(camera.getDistanceX(), camera.getDistanceY(), camera.getDistanceZ());
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

	const int numberOfThreads = 512; //arbituary value for number of threads, tbh this could be increased to 64,128 or 512 for faster processing
	const int numberOfBlocks = (N / numberOfThreads) + 1; //if more threads needed than can fit in a block, add another block

	rotateAndProject << <numberOfBlocks, numberOfThreads >> > (d_vectors, d_XYZoutput, h_width, N, camera);

	Vector* h_output = new Vector[N]; //the output of rotate and projecet (all vectors, except vectors in view of frustrum will be projected)
	cudaMemcpy(h_output, d_vectors, sizeof(Vector) * N, cudaMemcpyDeviceToHost);

	cudaFree(d_vectors);
	cudaFree(d_XYZoutput);

	return h_output;
}

__global__ void findTriangleOutlines(Vector* d_vectors, Pixel* d_pixel, const int N, const double maxX, const double maxY, const double rangeX, const double rangeY, const double width, const double height) {
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (i < N && d_vectors[i].getProjectVector() == true) {

		//get a vector + its connecting vector
		Vector vec1 = d_vectors[i];
		Vector vec2 = d_vectors[i + 1];
		if (i % 3 == 2) {
			vec2 = d_vectors[i - 3]; 
		}

		//the following 2 lines of code is already in the display class
		//although I would love to just call that function, its not saved on the gpu
		//making memory space for display is not a good idea; as it has SDL classes and functions
		vec1.x = ((vec1.x + maxX) / rangeX) * width;
		vec1.y = ((vec1.y + maxY) / rangeY) * height;
		vec2.x = ((vec2.x + maxX) / rangeX) * width;
		vec2.y = ((vec2.y + maxY) / rangeY) * height;

		//find line equation
		double m = (vec2.y - vec1.y) / (vec2.x - vec1.x);
		double c = vec1.y - (m * vec1.x);

		d_pixel[i].setValues(1, c, vec1.getB(), m, true);

		//find pixels which are outlined
		int x;
		double y;
		double pastY;
		int xBound;
		int yBound;
		double z;
		if (vec1.x < vec2.x) {
			int x = (int)vec1.x;
			double y = vec1.y;
			double pastY = vec1.y;
			xBound = (int)vec2.x;
			yBound = (int)vec2.y;
			z = vec1.z;
		}
		else {
			int x = (int)vec2.x;
			double y = vec2.y;
			double pastY = vec2.y;
			xBound = (int)vec1.x;
			yBound = (int)vec1.y;
			z = vec2.z;
		}
		

		//PROBLEM: 

		while ((int)x != xBound || (int)y != yBound) {
			break; //afdfdffkjfdjasdfjl;kfdjlk;adjflkadfkjjlkfdsjklafsedkjlfsddsaf;faffksdjkfdsfd;kfkjasfkafsdkasfdl;j
			x++;
			y = (m * x) + c;
			round(y);
			d_pixel[(int)y * (int)height + x].setValues(vec1.getR(), vec1.getG(), vec1.getB(), z, true);
			if (modulus<double>(pastY - y) > 1) {
				for (int j = pastY; j < (int)y; j++) {
					//outlines.add[x - 1, j];
					d_pixel[(int)j * (int)height + (x - 1)].setValues(vec1.getR(), vec1.getG(), vec1.getB(), z, true);
				}
			}
		}
		__syncthreads();
	}
}

//rasterisation, to be called from cpu file
__host__ Pixel* FindTriangleOutlines(Vector* h_vectors, const int N, const double maxX, const double maxY, const double rangeX, const double rangeY, const double width, const double height) {
	//h_vectors comprises of both projected and unprojected vectors
	//every 3 consecutive vectors in h_vectors is a triangle

	/**
		The amount of triangle outlines will vary each call to FindTriangleOutlines
		So it would make sense to use a dynamic data structure to store the triangle outlines
		After some research making a dynamic data structure for the GPU which can then transfer it's data across to the CPU could be a project itself

		To get around this problem, we will allocate memory space for each pixel on screen as an array
		When an outline is found, its saved into the array. If 50, 200 = red then array position [50,200] = red
		This 2d array of sorts will be represented in vector notation to avoid the errors which come with 2d arrays on the GPU

	*/

	Vector* d_vectors; //vectors on the device
 	Pixel* d_pixel; //stores data on every single pixel
	//array goes rightwards first, so across the width then one down

	const size_t sizeofVectors = sizeof(Vector) * N;
	const size_t sizeOfPixels = sizeof(Pixel) * width * height; 
	
	const int numberOfThreads = 128;
	const int numberOfBlocks = (N / numberOfThreads) + 1;

	cudaError error;

	error = cudaMalloc(&d_vectors, sizeofVectors);
	error = cudaMalloc(&d_pixel, sizeOfPixels);

	error = cudaMemcpy(d_vectors, h_vectors, sizeofVectors, cudaMemcpyHostToDevice);

	//findTriangleOutlines << <numberOfBlocks, numberOfThreads >> > (d_vectors, d_pixel, N, maxX, maxY, rangeX, rangeY, width, height);

	Pixel* h_pixel = new Pixel[height * width];
	error = cudaMemcpy(h_pixel, d_pixel, sizeOfPixels, cudaMemcpyDeviceToHost);

	int amountFilledIn = 0;

	for (int i = 0; i < width * height; i++) {
		if (h_pixel[i].isAnOutline() == true) {
			amountFilledIn++;
		}
		
	}

	cudaFree(d_vectors);
	cudaFree(d_pixel);

	return h_pixel;

}
