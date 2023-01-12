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
EngineDisplay engineDisplay(540, 960, "3D engine");
UIDisplay uiDisplay(700, 960, "UI display");


int main(int argc, char* args[]) {

	List<Vector> vecStore; //The store of vectors. Each 3 consecutive vectors form their own triangle.
	vecStore = loadDefaultShape(vecStore); //load the default shape
	addTriangleToVectorStore(vecStore, new Vector(2, 3, 5), new Vector(1, 2, 3), new Vector(3, 2, 5));
	
	for (int i = 0; i < 1000; i++) {
		addTriangleToVectorStore(vecStore, new Vector(i, i, i), new Vector(i + 1, i - 1, i), new Vector(i + 2, i, i));
	}
	
	//initilise camera
	Camera camera(0, 0, 0.3, (3.141592654 / 180) * 90, 50);

	//Load default color onto the screen
	engineDisplay.clearScreen();

	//initilise cuda:
	cudaFree(0);

	//initilise GPU fov values
	setUpFovValuesForGPU(camera.getFOVX(), engineDisplay.getHeight(), engineDisplay.getWidth());

	//initilise GPU cross product values for viewing frustrum (used for clipping)
	setUpCalculationForCrossProduct(engineDisplay.getMaxX(), engineDisplay.getMaxY(), engineDisplay.getRangeX(), engineDisplay.getRangeY(), camera.getFarClipDistanceZ());

	//initilise matrixes
	RotationMatrix xMatrix = initiliseXRotation();
	RotationMatrix yMatrix = initiliseYRotation(); 
	RotationMatrix zMatrix = initiliseZRotation();

	//initilise UI window
	initiliseUIWindow();
	uiDisplay.renderTextBoxes();
	uiDisplay.draw();

	//main game loop:
	SDL_Event event{}; //event handler
	const int lengthOfAFrame = 17; //how long a frame should last
	int frameTime = 0; //how long the last frame lasted
	
	const double howMuchToMove = 0.2; //how much the camera should move when a user inputs a movement
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
				

			//Because of preivous code in rotate and project, if one vector has been projected all connecting vectors would have also been projected
			//each 3 consecutive vectors in projectedVectors are connected, so can loop through every 3 vectors and check if it has been projected.
			//if it has, can pass that vector and each connecting one into a draw function
			engineDisplay.clearScreen(); //prepare the screen for drawing by getting rid of all current drawing on the sreen
			int howManyVectorsOnScreen = 0; //the amount of vectors displayed on screen. To be displayed onto the "amountOfVectorsOnScreen" textbox
			for (int i = 0; i < vecStore.count(); i += 3) {
				if (projectedVectors[i].getProjectVector() == true) {
					drawATriangle(projectedVectors[i], projectedVectors[i + 1], projectedVectors[i + 2]);
					howManyVectorsOnScreen += 3;
				}
			}

			engineDisplay.draw(); //draw the current scene

			frameTime = SDL_GetTicks() - frameTime;
			//update the UI display with new information
			uiDisplay.changeTextBasedOnName(std::to_string(frameTime), "frameTime");
			uiDisplay.changeTextBasedOnName(std::to_string(vecStore.count()), "amountOfVectorsInPlane");
			uiDisplay.changeTextBasedOnName(std::to_string(howManyVectorsOnScreen), "amountOfVectorsOnScreen");
			uiDisplay.clearScreen();
			uiDisplay.renderTextBoxes();
			uiDisplay.draw();
			if (frameTime < lengthOfAFrame) {
				SDL_Delay(frameTime);
			}
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

//vec store is passed by ref so dont have to return the whole thing
void addTriangleToVectorStore(List<Vector> &vecStore, Vector* vec1, Vector* vec2, Vector* vec3) {
	vecStore.add(*vec1);
	vecStore.add(*vec2);
	vecStore.add(*vec3);
}

//sets up values for the x rotation matrix
RotationMatrix initiliseXRotation() {
	double xRotationValues[] = { 1,0,0,0,1,1,0,-1,1 };
	int xRotationSinIndexes[] = { 5,7 };
	int xRotationCosIndexes[] = { 4,8 };
	return RotationMatrix(xRotationValues, xRotationSinIndexes, xRotationCosIndexes);

}

//setus up values for y rotation matrix
RotationMatrix initiliseYRotation() {
	double yRotationValues[] = { 1,0,-1,0,1,0,1,0,1 };
	int yRotationSinIndexes[] = { 2,6 };
	int yRotationCosIndexes[] = { 0,8 };
	return RotationMatrix(yRotationValues, yRotationSinIndexes, yRotationCosIndexes);
}

//sets up values for z rotation matrix
RotationMatrix initiliseZRotation() {
	double zRotationValues[] = { 1,1,0,-1,1,0,0,0,1 };
	int zRotationSinIndexes[] = { 1,3 };
	int zRotationCosIndexes[] = { 0,4 };
	return RotationMatrix(zRotationValues, zRotationSinIndexes, zRotationCosIndexes);

}

void initiliseUIWindow() {
	const std::string latoFilePath = "Lato-black.ttf"; //font files are saved in the solution
	const Uint8 r = 255;
	const Uint8 g = 255;
	const Uint8 b = 255;
	const int size1 = 12; //size of text. Uses normal like ms word sizes.

	const int heightOfText = 100;
	const int WidthOfText = uiDisplay.getWidth() / 1.75; // The height and width of text to be displayed

	const int startX = 10;
	const int startY = 10; // (10, 10) is located at the top left of the ui display window

	//The textboxes are identified by there name
	//If you want to update the text of a textbox, you pass the textbox's name into the update textbox function (along with the new text). This function is stored in UIDisplay
	uiDisplay.addTextbox(latoFilePath, "Time to Render A Frame:", "frameText", size1, startX, startY, WidthOfText, heightOfText, r, g, b); //Text time to render a frame
	uiDisplay.addTextbox(latoFilePath, "0", "frameTime", size1, startX + WidthOfText + 10, startY, WidthOfText / 7, heightOfText, r, g, b); //Displays how many ms the last frame was
	uiDisplay.addTextbox(latoFilePath, "Amount Of Vectors On Screen:", "amountOnScreenText", size1, startX, heightOfText + startY, WidthOfText, heightOfText, r, g, b); //Text Amount Of Vectors on Screen
	uiDisplay.addTextbox(latoFilePath, "0", "amountOfVectorsOnScreen", size1, startX + WidthOfText + 10, startY + heightOfText, WidthOfText / 7, heightOfText, r, g, b); //Displays the number of vectors on screen
	uiDisplay.addTextbox(latoFilePath, "Amount Of Vectors In Plane:", "amountOfVectorsInPlaneText", size1, startX, startY + heightOfText * 2, WidthOfText, heightOfText, r, g, b); //displays the text before the number of vectors in the plane
	uiDisplay.addTextbox(latoFilePath, "0", "amountOfVectorsInPlane", size1, startX + WidthOfText + 10, startY + heightOfText * 2, WidthOfText / 7, heightOfText, r, g, b); //Displays how many vectors are in the plane
	uiDisplay.addTextbox(latoFilePath, "RT to rotate Z axis, QE to rotate Y axis", "infoQERT", size1 - 2, startX, startY + heightOfText * 3, WidthOfText  * 1.3, heightOfText - 10, r, g, b); //information about the display
	uiDisplay.addTextbox(latoFilePath, "ZX to rotate X axis", "infoZX", size1 - 2, startX, startY + heightOfText * 4, WidthOfText - 100, heightOfText - 10, r, g, b); //information about x axis
	uiDisplay.addTextbox(latoFilePath, "WS for moving the Z axis, AD for moving X axis", "infoWSAD", size1 - 2, startX, startY + heightOfText * 5, WidthOfText * 1.5, heightOfText - 10, r, g, b); //info about moving bar UP DOWN 
	uiDisplay.addTextbox(latoFilePath, "UP DOWN arrow keys to move Y axis", "infoUPDOWN", size1 - 2, startX, startY  + heightOfText * 6, WidthOfText, heightOfText - 10, r, g, b); //info about moving up down
}