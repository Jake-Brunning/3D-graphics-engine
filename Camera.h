#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Camera
{
public:
	//near clip distances functions
	__host__ __device__ double getDistanceX() {
		return distanceX; 
	}

	__host__ __device__ double getDistanceY() {
		return distanceY;
	}

	__host__ __device__ double getDistanceZ() {
		return distanceZ;
	}

	__device__ double getFarClipDistanceZ() {
		return farClipZ;
	}

	//X angle functions
	void increaseRotationX(double incremeant) {
		rotatedX += incremeant;
	}

	double getRotatedX() {
		return rotatedX;
	}

	//y angle functions
	void increaseRotationY(double incremeant) {
		rotatedY += incremeant;
	}

	double getRotatedY() {
		return rotatedY;
	}

	//z angle functions
	void increaseRotationZ(double incremeant) {
		rotatedZ += incremeant;
	}

	double getRotatedZ() {
		return rotatedZ;
	}

	//FOVX functions
	double getFOVX() {
		return FOVX;
	}

	//constructors
	Camera(double distanceX, double distanceY, double distanceZ, double FOVX, double farClipZ) {
		this->distanceX = distanceX;
		this->distanceY = distanceY;
		this->distanceZ = distanceZ;
		this->FOVX = FOVX;
		this->farClipZ = farClipZ;
	}

private:
	//rotated is in radians
	//these is the angleX, angleY, angleZ values
	double rotatedX = 0;
	double rotatedY = 0;
	double rotatedZ = 0;

	//near clip distances
	double distanceX;
	double distanceY;
	double distanceZ;
	double farClipZ;
	double FOVX;
};

