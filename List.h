#pragma once
#pragma once
#include <iostream>;

template<typename type>
class List
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

private:
	type* data; //the store of data
	int NumOfElements = 0; //the amount of elements in the array
};

