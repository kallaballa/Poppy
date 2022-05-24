#ifndef SRC_SETTINGS_HPP_
#define SRC_SETTINGS_HPP_

#include <cstddef>
#include <unistd.h>

namespace poppy {
class Settings {
private:
	static Settings* instance_;
	Settings() {};
public:
	bool show_gui = false;
	double number_of_frames = 60;
	double contour_sensitivity = 1;
	double match_tolerance = 5;
	off_t max_keypoints = -1;
	size_t pyramid_levels = 4;
	bool enable_auto_align = false;
	bool enable_radial_mask = false;
	bool enable_face_detection = false;
	bool enable_denoise = false;
	bool enable_src_scaling = false;

	static Settings& instance() {
		if(instance_ == nullptr)
			instance_ = new Settings();

		return *instance_;
	}
};
}
#endif
