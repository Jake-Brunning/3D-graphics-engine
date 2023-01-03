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
void drawATriangle(Vector vec1, Vector vec2, Vector vec3);
void initiliseUIWindow();
RotationMatrix initiliseXRotation();
RotationMatrix initiliseYRotation();
RotationMatrix initiliseZRotation();

//Variable declarations
EngineDisplay engineDisplay(1080, 1920, "3D engine");
UIDisplay uiDisplay(540, 960, "UI display");


int main(int argc, char* args[]) {

	List<Vector> vecStore; //The store of vectors. Each 3 consecutive vectors form their own triangle.
	vecStore = loadDefaultShape(vecStore); //load the default shape
	//addTriangleToVectorStore(vecStore, new Vector(2, 3, 5), new Vector(1, 2, 3), new Vector(3, 2, 5));


	//initilise camera
	Camera camera(0, 0, 0.3, (3.141592654 / 180) * 120);
	 
	//Load default color onto the screen
	engineDisplay.clearScreen();

	//initilise cuda:
	cudaFree(0);

	//initilise GPU fov values
	setUpFovValuesForGPU(camera.getFOVX(), engineDisplay.getHeight(), engineDisplay.getWidth());

	//initilise matrixes
	RotationMatrix xMatrix = initiliseXRotation();
	RotationMatrix yMatrix = initiliseXRotation(); 
	RotationMatrix zMatrix = initiliseZRotation();

	//main game loop:
	SDL_Event event{}; //event handler
	const int lengthOfAFrame = 17; //how long a frame should last
	int frameTime = 0; //how long the last frame lasted
	
	const double howMuchToMove = 0.02; //how much the camera should move when a user inputs a movement
	const double howMuchToRotate = 0.0314; //how much the camera should rotate when a user inputs a movement
	
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
			camera.increaseRotationX(-howMuchToRotate);
			break;
		case SDLK_e:
			//rotate X right
			camera.increaseRotationX(howMuchToRotate);
			break;
		case SDLK_z:
			//rotate Y left
			camera.increaseRotationY(-howMuchToRotate);
			break;
		case SDLK_x:
			//rotate Y right
			camera.increaseRotationY(howMuchToRotate);
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

		if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE){ //handles closing of the windows
			break;
		}
		
		if (eventHappened) {
			//projected vectors consists of all vectors. The vectors in the view of the frustrum have been projected; ones which are not have not been projected
			//This CREATES A COPY of vecstore
			Vector* projectedVectors = setUpRotationAndProjection(xMatrix.setUpData(camera.getRotatedX()), yMatrix.setUpData(camera.getRotatedY()), zMatrix.setUpData(camera.getRotatedZ()), vecStore.changeToArray(), vecStore.count(), camera);
				
			//Pixel* triOutlines = FindTriangleOutlines(projectedVectors, vecStore.count(), engineDisplay.getMaxX(), engineDisplay.getMaxY(), engineDisplay.getRangeX(), engineDisplay.getRangeY(), engineDisplay.getWidth(), engineDisplay.getHeight());
			
			//Because of prievous code in rotate and project, if one vector has been projected all connecting vectors would have also been projected
			//each 3 consecutive vectors in projectedVectors are connected, so can loop through every 3 vectors and check if it has been projected.
			//if it has, can pass that vector and each connecting one into a draw function
			engineDisplay.clearScreen(); //prepare the screen for drawing by getting rid of all current drawing on the sreen
			for (int i = 0; i < vecStore.count(); i += 3) {
				if (projectedVectors[i].getProjectVector() == true) {
					drawATriangle(projectedVectors[i], projectedVectors[i + 1], projectedVectors[i + 2]);
				}
			}

			engineDisplay.draw(); //draw the current scene
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


//draw call for a triangle
void drawATriangle(Vector vec1, Vector vec2, Vector vec3) {
	engineDisplay.renderLine(vec1.x, vec2.x, vec1.y, vec2.y);
	engineDisplay.renderLine(vec2.x, vec3.x, vec2.y, vec3.y);
	engineDisplay.renderLine(vec3.x, vec1.x, vec3.y, vec1.y);
}



//the first shape to be loaded onto the program
List<Vector> loadDefaultShape(List<Vector> vecstore) {
	Vector* vec1 = new Vector(1.5, 2, 7);
	Vector* vec2 = new Vector(2, 1, 7);
	Vector* vec3 = new Vector(1.2, 1, 7);
	addTriangleToVectorStore(vecstore, vec1, vec2, vec3);
	return vecstore;
}

void addTriangleToVectorStore(List<Vector> &vecStore, Vector* vec1, Vector* vec2, Vector* vec3) {
	vecStore.add(*vec1);
	vecStore.add(*vec2);
	vecStore.add(*vec3);
}


RotationMatrix initiliseXRotation() {
	double xRotationValues[] = { 1,0,0,0,1,1,0,-1,1 };
	int xRotationSinIndexes[] = { 5,7 };
	int xRotationCosIndexes[] = { 4,8 };
	return RotationMatrix(xRotationValues, xRotationSinIndexes, xRotationCosIndexes);

}

RotationMatrix initiliseYRotation() {
	double yRotationValues[] = { 1,0,-1,0,1,0,1,0,1 };
	int yRotationSinIndexes[] = { 2,6 };
	int yRotationCosIndexes[] = { 0,8 };
	return RotationMatrix(yRotationValues, yRotationSinIndexes, yRotationCosIndexes);
}

RotationMatrix initiliseZRotation() {
	double zRotationValues[] = { 1,1,0,-1,1,0,0,0,1 };
	int zRotationSinIndexes[] = { 1,3 };
	int zRotationCosIndexes[] = { 0,4 };
	return RotationMatrix(zRotationValues, zRotationSinIndexes, zRotationCosIndexes);

}

void initiliseUIWindow() {
	const std::string fontFilePath;
}