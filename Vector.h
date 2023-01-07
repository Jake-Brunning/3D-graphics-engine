#pragma once
#include "cuda_runtime.h";
#include "device_launch_parameters.h"
class Vector
{
public:
	double x = 0;
	double y = 0;
	double z = 0;

	//constructors
	__host__ __device__ Vector(double x, double y, double z) {
		this->x = x;
		this->y = y;
		this->z = z;
		
	}

	Vector(double x, double y, double z, short r, short g, short b) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->r = r;
		this->g = g;
		this->b = b;
	}

	Vector() {
		//No parameter constructor so list class can make memory space without
		//needing to initilise any values
	}

	//GPU project function
	__device__ void projectVector(double nearClipDistX, double nearClipDistY, double nearClipDistZ) {
		projected = true;
		if (z == 0) {
			x = nearClipDistX;
			y = nearClipDistY;
			return;
		}
		else if (z < 0) {
			z = z * -1;
		}

		x = ((nearClipDistZ / z) * x) + nearClipDistX;
		y = ((nearClipDistZ / z) * y) + nearClipDistY;
	}

	__device__ bool getProjectVector() {
		return projected;
	}

	//get functions for R G B
	__device__ __host__ short getR() {
		return r;
	}

	__device__ __host__ short getB() {
		return b;
	}

	__device__ __host__ short getG() {
		return g;
	}

	//set functions for R G B
	void setR(short r) {
		this->r = r;
	}

	void setB(short b) {
		this->b = b;
	}

	void setG(short g) {
		this->g = g;
	}

private:
	short r = 0;
	short b = 0;
	short g = 255;
	bool projected = false; //true means the vector has been projected

};

