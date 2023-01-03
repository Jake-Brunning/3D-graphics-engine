 #pragma once
#pragma once
#include <iostream>;
#include "cuda_runtime.h";
#include "device_launch_parameters.h"
template<typename type>
class List //List class to be used on the CPU
{
public:

	List(int startLength = 0) : NumOfElements(startLength) {
		data = new type[NumOfElements];
	}

	void add(type data) {
		//Create space for new data
		NumOfElements++;
		type* tempArray = new type[NumOfElements];

		//put old data into new data store
		for (int i = 0; i < NumOfElements; i++) {
			tempArray[i] = this->data[i];
		}

		//add the new data to the new data store
		tempArray[NumOfElements - 1] = data;

		//rewrite old data with new data
		this->data = tempArray;
	}

	type getIndex(int index) {
		return data[index];
	}

	int count() {
		return NumOfElements;
	}

	//convert to array
	type* changeToArray() {
		return data;
	}

	//convert to list
	void changetolist(type* data) {
		this->data = data;
		int NumOfElements = sizeof(*data) / sizeof(type);
	}

private:
	type* data; //the store of data
	int NumOfElements = 0; //the amount of elements in the array
};

