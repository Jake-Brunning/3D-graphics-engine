#include "cuda_runtime.h";
#include "device_launch_parameters.h"
#include "Vector.h"

extern Vector* setUpMoveVectors(double changeInXYZ, char axis, Vector* vectors, int N);