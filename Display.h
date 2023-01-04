#pragma once
#include <SDL.h>
#include <SDL_ttf.h> //external library source: https://github.com/libsdl-org/SDL_ttf
#include <iostream>
#include <string>


#include "List.h"
class BaseDisplay {
public:
	//Constructor and deconstructor is sourced from the SDL documentation
	BaseDisplay(int height_, int width_, std::string title) : height(height_), width(width_) {

		if (SDL_Init(SDL_INIT_VIDEO) < 0) {
			std::cout << "SDL could not initialize! SDL error: " << SDL_GetError() << std::endl;
		}

		window = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
		renderer = SDL_CreateRenderer(window, 0, SDL_WINDOW_SHOWN);
		int x = TTF_Init();

	}
	~BaseDisplay() {
		SDL_DestroyWindow(window);
		SDL_DestroyRenderer(renderer);
		SDL_Quit();
	}

	void clearScreen(int r = 0, int g = 0, int b = 0) {
		SDL_SetRenderDrawColor(renderer, r, g, b, 0);
		SDL_RenderClear(renderer);
		SDL_RenderPresent(renderer);
	}

	void draw() {
		SDL_RenderPresent(renderer);
	}

	int getWidth() {
		return width;
	}

	int getHeight() {
		return height;
	}

protected:
	SDL_Window* window = NULL;
	SDL_Renderer* renderer = NULL;
	const int height; //height of the display in pixels
	const int width; //width of the display in pixels
};

class EngineDisplay : public BaseDisplay {
public:

	using BaseDisplay::BaseDisplay;

	void ConvertToPixelCoordinates(double& x, double& y) {
		x = ((x + maxX) / rangeX) * width;
		y = ((y + maxY) / rangeY) * height;
	}

	void renderLine(double x1, double x2, double y1, double y2, int r = 0, int g = 255, int b = 0) {
		ConvertToPixelCoordinates(x1, y1);
		ConvertToPixelCoordinates(x2, y2);
		SDL_SetRenderDrawColor(renderer, r, g, b, 0);
		SDL_RenderDrawLine(renderer, x1, y1, x2, y2);
	}

	double getMaxX() {
		return maxX;
	}

	double getMaxY() {
		return maxY;
	}

	double getRangeX() {
		return rangeX;
	}

	double getRangeY() {
		return rangeY;
	}

private:
	const double maxX = 1; //max value for x for normalised plane
	const double maxY = 1; //max value for y for normalised plane
	const double rangeX = 2; //max range for normalised plane (x)
	const double rangeY = 2; //max range for normalised plane (y)
};

class UIDisplay : public BaseDisplay {
public:
	UIDisplay(int height_, int width_, std::string title) : BaseDisplay(height_, width_, title) {
		TTF_Init(); // need to initilse the text to texture before using it, so have to make a new construcutor
		//the base constructor is still called
	}

	void renderTextBoxes() {
		for (int i = 0; i < textBoxes.count(); i++) { //go through each textbox and add its texture to the renderer
			SDL_RenderCopy(renderer, textBoxes.getIndex(i)->getTexture(), NULL, textBoxes.getIndex(i)->getRect());
		}
	}

	void changeTextBasedOnName(const std::string newText, const std::string name) { //changes and updates the texture of the target text box
		for (int i = 0; i < textBoxes.count(); i++) {
			if (textBoxes.getIndex(i)->getName() == name) {
				textBoxes.getIndex(i)->changeText(&renderer, newText);
			}
			
		}
	}

	void addTextbox(std::string fontFilePath, std::string text, std::string name, int size, int x, int y, int w, int h, Uint8 r = 0, Uint8 g = 0, Uint8 b = 0) {//add a textbox 
		textBoxes.add(new TextBox(fontFilePath, text, name, size, x, y, w, h, &renderer, r, g, b));
	}

private:
	class TextBox { //purpose is just to display text. The user cannot interact with instances of this class
	public:
		TextBox(std::string fontFilePath, std::string text, std::string name, int size, int x, int y, int w, int h, SDL_Renderer** renderer, Uint8 r, Uint8 g, Uint8 b) {
			this->name = name;

			rect->x = x;
			rect->y = y;
			rect->h = h;
			rect->w = w;

			color = { r, g, b };

			font = TTF_OpenFont(fontFilePath.c_str(), size); //create font
			textStore = TTF_RenderText_Solid(font, text.c_str(), color); //use font to chagne string into displayable text
			textToDisplay = SDL_CreateTextureFromSurface(*renderer, textStore); //convert text to a texture
		}
		~TextBox() { //deconstructor to free memory space
			SDL_FreeSurface(textStore);
			SDL_DestroyTexture(textToDisplay);
			TTF_CloseFont(font);
		}

		void changeText(SDL_Renderer** renderer, const std::string newText) {
			textStore = TTF_RenderText_Solid(font, newText.c_str(), color); //update the text
			textToDisplay = SDL_CreateTextureFromSurface(*renderer, textStore); //update the texture
		}

		SDL_Texture* getTexture() {
			return textToDisplay;
		}

		SDL_Rect* getRect() {
			return rect;
		}

		std::string getName() {
			return name;
		}

	private:
		SDL_Rect* rect = new SDL_Rect; //rectangle which will contain the text
		SDL_Color color; //the color of the text
		TTF_Font* font; //the font of the text
		SDL_Surface* textStore; //formats the text using font and color, and makes it convertable to a texture
		SDL_Texture* textToDisplay; //The texture which will be displayed to the screen
		std::string name; //name of the button
	};

	List<TextBox*> textBoxes; //list of all the textboxes on the display
};