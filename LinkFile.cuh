#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vector.h"
#include "Camera.h"
#include "Pixel.h"

extern Vector* setUpMoveVectors(double changeInXYZ, char axis, Vector* vectors, int N);
extern Vector* setUpRotationAndProjection(double xRotation[9], double yRotation[9], double zRotation[9], Vector* h_vectors, int N, const Camera camera);
extern void setUpFovValuesForGPU(double zDistFromNearClip, double yPixels, double xPixels);
extern void setUpCalculationForCrossProduct(const double maxX, const double maxY, const double rangeX, const double rangeY, const double zFarDist);