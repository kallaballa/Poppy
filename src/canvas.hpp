#ifndef CANVAS_H_
#define CANVAS_H_

#include <cwchar>
#include <SDL/SDL.h>
#include <cstdint>

typedef double fd_float_t;
typedef int64_t fd_coord_t;
typedef uint64_t fd_dim_t;
typedef uint32_t fd_image_pix_t;
constexpr int FD_IMAGE_DEPTH_IN_BYTES = 4;
typedef uint32_t fd_iter_count_t;
typedef fd_image_pix_t* image_t;

namespace poppy {
class Canvas {
private:
	fd_dim_t width_;
	fd_dim_t height_;
	class SDL_Surface *screen_;
	bool offscreen_;
	const int BYTES_PER_PIXEL = FD_IMAGE_DEPTH_IN_BYTES;
public:
	Canvas(const fd_dim_t& width, const fd_dim_t& height, const bool& offscreen = false);
	virtual ~Canvas() {
	}
	void flip() volatile;
	void draw(image_t const& image) volatile;
};
}
#endif /* CANVAS_H_ */
