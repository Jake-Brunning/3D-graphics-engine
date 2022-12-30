#include <SDL.h>
#include <stdio.h>
#include <iostream>

#include "LinkFile.cuh";
#include "Camera.h";
#include "Display.h";
#include "Vector.h";
#include "List.h";

//function declarations
List<Vector> loadDefaultShape(List<Vector> vecstore);
void addTriangleToVectorStore(List<Vector>& vecStore, Vector* vec1, Vector* vec2, Vector* vec3);


//Variable declarations
Display engineDisplay(700, 900, "3D engine");

int main(int argc, char* args[]) {
	List<Vector> vecStore; //The store of vectors
	vecStore = loadDefaultShape(vecStore); //load the default shape
	
	Camera camera(0, 0, 0.5, 90);

	//Load default color onto the screen
	engineDisplay.clearScreen();

	//initilise cuda:
	cudaFree(0);

	//main game loop:
	SDL_Event event{}; //event handler
	const int lengthOfAFrame = 17; //how long a frame should last
	int frameTime = 0; //how long the last frame lasted
	
	const double howMuchToMove = 0.2; //how much the camera should move when a user inputs a movement
	

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
			camera.increaseRotationX(0.314);
			break;
		case SDLK_e:
			//rotate X right
			camera.increaseRotationX(-0.314);
			break;
		case SDLK_z:
			//rotate Y left
			camera.increaseRotationY(-0.314);
			break;
		case SDLK_x:
			//rotate Y right
			camera.increaseRotationX(0.314);
			break;
		case SDLK_r:
			//rotate Z left
			camera.increaseRotationZ(-0.314);
			break;
		case SDLK_t:
			//rotate Z right
			camera.increaseRotationZ(0.314);
			break;
		}

		if (event.type == SDL_QUIT) {
			break;
		}
		
		if (event.type != NULL) { //if an event has happend
			//draw and project vectors
		}

		frameTime = SDL_GetTicks() - frameTime;
		if (frameTime < lengthOfAFrame) {
			SDL_Delay(frameTime);
		}
		SDL_PollEvent(&event);
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