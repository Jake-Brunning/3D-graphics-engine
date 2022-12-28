#pragma once
#include "cuda_runtime.h";
#include "device_launch_parameters.h"
#include "CudaLinkTest.cuh"
class Vector
{
public:
	double x = 0;
	double y = 0;
	double z = 0;

	//varying constructors which coudl be used
	Vector(double x, double y, double z) {
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

	//assign the connected vectors
	void assignConnectedVectors(Vector* vec2, Vector* vec3) {
		this->vec2 = vec2;
		this->vec3 = vec3;
	}

	//GPU project function
	__device__ void projectVector(double nearClipDistX, double nearClipDistY, double nearClipDistZ) {
		x = ((nearClipDistZ / z) * x) + nearClipDistX;
		y = ((nearClipDistZ / z) * y) + nearClipDistY;
	}

private:
	Vector* vec2 = nullptr;
	Vector* vec3 = nullptr;

	short r = 0;
	short b = 0;
	short g = 255;

	//get functions for R G B
	short getR() {
		return r;
	}
	
	short getB() {
		return b;
	}

	short getG() {
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
};
