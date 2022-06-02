#include <cassert>
#include <iostream>
#include "canvas.hpp"

namespace poppy {
Canvas::Canvas(const fd_dim_t& width, const fd_dim_t& height, const bool& offscreen) :
		width_(width), height_(height), screen_(NULL), offscreen_(offscreen) {

	if (width > 0 && height > 0) {
		if (SDL_Init(SDL_INIT_VIDEO) == -1) {
			std::cerr << "Can't init SDL: " + std::string(SDL_GetError()) << std::endl;
			exit(1);
		}
		atexit(SDL_Quit);

		if (!offscreen) {
			screen_ = SDL_SetVideoMode(width, height, BYTES_PER_PIXEL * 8, SDL_SWSURFACE | SDL_RESIZABLE);
		} else
			screen_ = SDL_CreateRGBSurface(SDL_SWSURFACE, width, height, BYTES_PER_PIXEL * 8, 0, 0, 0, 0);

		if (screen_ == NULL) {
			std::cerr << "Can't set video mode: " + std::string(SDL_GetError()) << std::endl;
			exit(1);
		}
	}
}

void Canvas::resize(size_t width, size_t height) volatile {
	screen_ = SDL_SetVideoMode(width, height, BYTES_PER_PIXEL * 8, SDL_SWSURFACE | SDL_RESIZABLE);
	width_ = width;
	height_ = height;

}

std::pair<fd_dim_t, fd_dim_t> Canvas::getSize() volatile {
	return {width_, height_};
}
void Canvas::flip() volatile {
	if (!offscreen_) {
		SDL_Flip(screen_);
	}
}

void Canvas::draw(image_t const& image) volatile {
	if (SDL_MUSTLOCK(screen_))
		SDL_LockSurface(screen_);

	memcpy(static_cast<void*>(screen_->pixels), static_cast<void*>(image), width_ * height_ * sizeof(fd_image_pix_t));
	flip();

	if (SDL_MUSTLOCK(screen_))
		SDL_UnlockSurface(screen_);
}

}
