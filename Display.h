#pragma once
#include <SDL.h>
#include <iostream>
#include <string>

class Display {
public:
	//Constructor and deconstructor is sourced from the SDL documentation
	Display(int height_, int width_, std::string title) : height(height_), width(width_) { 

		if (SDL_Init(SDL_INIT_VIDEO) < 0) {
			std::cout << "SDL could not initialize! SDL error: " << SDL_GetError() << std::endl;
		}

		window = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
		renderer = SDL_CreateRenderer(window, 0, SDL_WINDOW_SHOWN);

	}
	~Display() {
		SDL_DestroyWindow(window);
		SDL_DestroyRenderer(renderer);
		SDL_Quit();
	}

	void clearScreen(int r = 0, int g = 0, int b = 0) {
		SDL_SetRenderDrawColor(renderer, r, g, b, 0);
		SDL_RenderClear(renderer);
		SDL_RenderPresent(renderer);
	}

	void renderLine(double x1, double x2, double y1, double y2, int r = 0, int g = 255, int b = 0) {
		ConvertToPixelCoordinates(x1, y1);
		ConvertToPixelCoordinates(x2, y2);
		SDL_SetRenderDrawColor(renderer, r, g, b, 0);
		SDL_RenderDrawLine(renderer, x1, y1, x2, y2);
	}

	void ConvertToPixelCoordinates(double&x, double&y) {
		x = ((x + maxX) / rangeX) * width;
		y = ((y + maxY) / rangeY) * height;
	}

	void draw() {
		SDL_RenderPresent(renderer);
	}

	int getHeight() {
		return height;
	}

	int getWidth() {
		return width;
	}

private:
	SDL_Window* window = NULL;
	SDL_Renderer* renderer = NULL;
	const int height; //height of the display in pixels
	const int width; //width of the display in pixels
	const double maxX = 1; //max value for x for normalised plane
	const double maxY = 1; //max value for y for normalised plane
	const double rangeX = 2; //max range for normalised plane (x)
	const double rangeY = 2; //max range for normalised plane (y)
};