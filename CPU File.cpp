#include <SDL.h>
#include <stdio.h>
#include <iostream>

#include "LinkFile.cuh";
#include "Camera.h";
#include "Display.h";
#include "Vector.h";
#include "List.h";
#include "RotationMatrix.h";

//function declarations
List<Vector> loadDefaultShape(List<Vector> vecstore);
void addTriangleToVectorStore(List<Vector>& vecStore, Vector* vec1, Vector* vec2, Vector* vec3);


//Variable declarations
Display engineDisplay(700, 900, "3D engine");

int main(int argc, char* args[]) {

	List<Vector> vecStore; //The store of vectors
	vecStore = loadDefaultShape(vecStore); //load the default shape
	
	//initilise camera
	Camera camera(0, 0, 0.5, 90);

	//Load default color onto the screen
	engineDisplay.clearScreen();

	//initilise cuda:
	cudaFree(0);

	//initilise GPU fov values
	setUpFovValuesForGPU(camera.getFOVX(), engineDisplay.getHeight(), engineDisplay.getWidth());

	//initilise matrixes
	double xRotationValues[] = { 1,0,0,0,1,1,0,-1,1 };
	int xRotationSinIndexes[] = { 5,7 };
	int xRotationCosIndexes[] = { 4,8 };
	RotationMatrix xMatrix(xRotationValues, xRotationSinIndexes, xRotationCosIndexes);

	double yRotationValues[] = { 1,0,-1,0,1,0,1,0,1 };
	int yRotationSinIndexes[] = { 2,6 };
	int yRotationCosIndexes[] = { 0,8 };
	RotationMatrix yMatrix(yRotationValues, yRotationSinIndexes, yRotationCosIndexes);

	double zRotationValues[] = {1,1,0,-1,1,0,0,0,1};
	int zRotationSinIndexes[] = { 1,3 };
	int zRotationCosIndexes[] = { 0,4 };
	RotationMatrix zMatrix(zRotationValues, zRotationSinIndexes, zRotationCosIndexes);

	//main game loop:
	SDL_Event event{}; //event handler
	const int lengthOfAFrame = 17; //how long a frame should last
	int frameTime = 0; //how long the last frame lasted
	
	const double howMuchToMove = 0.2; //how much the camera should move when a user inputs a movement
	const double howMuchToRotate = 0.314; //how much the camera should rotate when a user inputs a movement
	
	bool eventHappened = true; //is true if an event has happened

	while (true)
	{
		frameTime = SDL_GetTicks();

		//go through user input 
		switch (event.key.keysym.sym) {
		//camera movements
		case SDLK_w:
			//move Z vectors backwards
			vecStore.changetolist(setUpMoveVectors(-howMuchToMove, 'z', vecStore.changeToArray(), vecStore.count()));
			break;
		case SDLK_a:
			//move X vectors right
			vecStore.changetolist(setUpMoveVectors(howMuchToMove, 'x', vecStore.changeToArray(), vecStore.count()));
			break;
		case SDLK_d:
			//move X vectors left
			vecStore.changetolist(setUpMoveVectors(-howMuchToMove, 'x', vecStore.changeToArray(), vecStore.count()));
			break;
		case SDLK_s:
			//move Z vectors forwards
			vecStore.changetolist(setUpMoveVectors(howMuchToMove, 'z', vecStore.changeToArray(), vecStore.count()));
			break;
		case SDLK_UP:
			//move Y vectors up
			vecStore.changetolist(setUpMoveVectors(howMuchToMove, 'y', vecStore.changeToArray(), vecStore.count()));
			break;
		case SDLK_DOWN:
			//move Y vectors down
			vecStore.changetolist(setUpMoveVectors(-howMuchToMove, 'y', vecStore.changeToArray(), vecStore.count()));
			break;

		//camera rotations
		case SDLK_q:
			//rotate X left
			camera.increaseRotationX(howMuchToRotate);
			break;
		case SDLK_e:
			//rotate X right
			camera.increaseRotationX(-howMuchToRotate);
			break;
		case SDLK_z:
			//rotate Y left
			camera.increaseRotationY(-howMuchToRotate);
			break;
		case SDLK_x:
			//rotate Y right
			camera.increaseRotationX(howMuchToRotate);
			break;
		case SDLK_r:
			//rotate Z left
			camera.increaseRotationZ(-howMuchToRotate);
			break;
		case SDLK_t:
			//rotate Z right
			camera.increaseRotationZ(howMuchToRotate);
			break;
		default:
			//if no inputs have occured
			eventHappened = false;
			break;
		}

		if (event.type == SDL_QUIT) {
			break;
		}
		
		if (eventHappened) {
			setUpRotationAndProjection(xMatrix.setUpData(camera.getRotatedX()), yMatrix.setUpData(camera.getRotatedY()), zMatrix.setUpData(camera.getRotatedZ()), vecStore.changeToArray(), vecStore.count());
		}

		frameTime = SDL_GetTicks() - frameTime;
		if (frameTime < lengthOfAFrame) {
			SDL_Delay(frameTime);
		}
		SDL_PollEvent(&event);
		eventHappened = true;
	}

	return 10;
}



//the first shape to be loaded onto the program
List<Vector> loadDefaultShape(List<Vector> vecstore) {
	Vector* vec1 = new Vector(0.5, 0.5, 2);
	Vector* vec2 = new Vector(0.5, 0.1, 2);
	Vector* vec3 = new Vector(0.3, 0.1, 2);
	addTriangleToVectorStore(vecstore, vec1, vec2, vec3);
	return vecstore;
}

void addTriangleToVectorStore(List<Vector> &vecStore, Vector* vec1, Vector* vec2, Vector* vec3) {
	vecStore.add(*vec1);
	vecStore.add(*vec2);
	vecStore.add(*vec3);
}