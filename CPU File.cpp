#include <SDL.h>
#include <stdio.h>
#include <iostream>
#include "CudaLinkTest.cuh";

#include "Camera.h";
#include "Display.h";
#include "Vector.h";

Display engineDisplay(500, 500, "3D engine");

int main(int argc, char* args[]) {
	std::cout << "Hello world" << std::endl;
	cpuEnterFunction();
	return 1;
}