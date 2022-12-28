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

	int getHeight() {
		return height;
	}

	int getWidth() {
		return width;
	}

private:
	SDL_Window* window = NULL;
	SDL_Renderer* renderer = NULL;
	const int height;
	const int width;
};