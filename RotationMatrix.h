#pragma once
#include <math.h>;
class RotationMatrix
{
public:
	
	RotationMatrix(double data[9], int indexesWhereSinIs[2], int indexesWhereCosIs[2]) { //+trig function should be saved as 1, -trig function should be saved as -1
		for (int i = 0; i < row * column; i++) {
			this->data[i] = data[i];
		}

		this->indexesWhereSinIs[0] = indexesWhereSinIs[0];
		this->indexesWhereSinIs[1] = indexesWhereSinIs[1];

		this->indexesWhereCosIs[0] = indexesWhereCosIs[0];
		this->indexesWhereCosIs[1] = indexesWhereCosIs[1];
	}

	double* setUpData(double theta, int N = 9) { //theta is an angle in radians
		//this function changes the 1s and 0s saved where sin and cos should be located
		//to sin(theta) and cos(theta)
		//and copys that array to the return

		double* arrToReturn = copyArray<double>(N);

		arrToReturn[indexesWhereCosIs[0]] = cos(theta) * data[indexesWhereCosIs[0]];
		arrToReturn[indexesWhereCosIs[1]] = cos(theta) * data[indexesWhereCosIs[1]];
		arrToReturn[indexesWhereSinIs[0]] = sin(theta) * data[indexesWhereSinIs[0]];
		arrToReturn[indexesWhereSinIs[1]] = sin(theta) * data[indexesWhereSinIs[1]];

		//in order to make trig function + or - a +1 or -1 is saved where the trig function is located
		//and trig function is multiplied by that value

		return arrToReturn;
	}


private:
	double data[9]; //It would make alot more sense if data was a 2d array (so it would look more like a matrix)
	//however, using 2d arrays with the gpu causes alot of issues
	//mainly with allocating memory, and then copying the memory from CPU to GPU and GPU to CPU, to the point where
	//an employee from NVIDA who worked on the cuda framework suggested to completly avoid 2d arrays if possible


	const int row = 3; //rows of data 2d array
	const int column = 3; ///columns of data 2d array
	int indexesWhereCosIs[2];
	int indexesWhereSinIs[2];

	template<typename type>
	type* copyArray(int N = 0) {
		type* returnArr = new type[N];

		for (int i = 0; i < N; i++) {
			returnArr[i] = data[i];
		}

		return returnArr;
	}
};

