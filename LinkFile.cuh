#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vector.h"
#include "Camera.h"

extern Vector* setUpMoveVectors(double changeInXYZ, char axis, Vector* vectors, int N);
extern Vector* setUpRotationAndProjection(double xRotation[9], double yRotation[9], double zRotation[9], Vector* h_vectors, int N, const Camera camera);
extern void setUpFovValuesForGPU(double fovInput, double yPixels, double xPixels);