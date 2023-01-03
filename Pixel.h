#pragma once
#include "cuda_runtime.h";
#include "device_launch_parameters.h"
class Pixel
{
	//Class saves data about a pixel on screen
	//its x and y values will be gotten from its position in the array which stores all pixels
public:
	//get functions for r g b
	//only host code sets their values, so functions only saved on the CPU
	short getRValue() {
		return r;
	}

	short getBValue() {
		return b;
	}

	short getGValue() {
		return g;
	}

	bool isAnOutline() {
		return coloured;
	}

	__device__ void setValues(short r, short b, short g, double z, bool isColoured) {
		this->r = r;
		this->b = b;
		this->g = g;
		this->z = z;
		this->coloured = isColoured;
	}

private:
	short r = 0;
	short b = 0;
	short g = 0;
	double z = 0;
	bool coloured = false; //indicates if the pixel will be coloured in or  not
};

