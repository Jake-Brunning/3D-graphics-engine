#pragma once
class Camera
{
public:
	//near clip distances functions
	double getDistanceX() {
		return distanceX; 
	}

	double getDistanceY() {
		return distanceY;
	}

	double getDistanceZ() {
		return distanceZ;
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
	Camera(double distanceX, double distanceY, double distanceZ, double FOVX) {
		this->distanceX = distanceX;
		this->distanceY = distanceY;
		this->distanceZ = distanceZ;
		this->FOVX = FOVX;
	}

private:
	//rotated is in radians
	//these is the angleX, angleY, angleZ values
	double rotatedX = 0;
	double rotatedY = 0;
	double rotatedZ = 0;

	//near clip distances
	double distanceX = 0;
	double distanceY = 0;
	double distanceZ = 1;
	double FOVX = 90;
};

