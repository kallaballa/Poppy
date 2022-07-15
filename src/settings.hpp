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
	bool enable_wait = false;
	double number_of_frames = 60;
	double frame_rate = 30;
	double match_tolerance = 1;
	size_t pyramid_levels = 64;
	bool enable_auto_align = false;
	bool enable_radial_mask = false;
	bool enable_face_detection = false;
	bool enable_denoise = false;
	bool enable_src_scaling = false;
	size_t face_neighbors = 6;
	std::string fourcc = "FFV1";

	static Settings& instance() {
		if(instance_ == nullptr)
			instance_ = new Settings();

		return *instance_;
	}
};
}
#endif
