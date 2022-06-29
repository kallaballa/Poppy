#ifndef SRC_SETTINGS_HPP_
#define SRC_SETTINGS_HPP_

#include <cstddef>
#include <unistd.h>
#include <string>

namespace poppy {
class Settings {
private:
	static Settings* instance_;
	Settings() {};
public:
	bool show_gui = false;
	double number_of_frames = 60;
	double frame_rate = 30;
	double contour_sensitivity = 1;
	double match_tolerance = 1;
	off_t max_keypoints = -1;
#ifndef _WASM
	size_t pyramid_levels = 64;
#else
	size_t pyramid_levels = 32;
#endif
	bool enable_auto_align = false;
	bool enable_radial_mask = false;
	bool enable_face_detection = false;
	bool enable_denoise = false;
	bool enable_src_scaling = false;
	std::string fourcc = "FFV1";

	static Settings& instance() {
		if(instance_ == nullptr)
			instance_ = new Settings();

		return *instance_;
	}
};
}
#endif
