#ifndef SRC_POPPY_HPP_
#define SRC_POPPY_HPP_

#include "util.hpp"
#include "algo.hpp"
#include "settings.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace poppy {
double ease_in_out_cubic(double x) {
	return ((x < 0.5 ? 4 * x * x * x : 1 - pow(-2 * x + 2, 3) / 2));
}
void init(bool showGui, size_t numberOfFrames, double matchTolerance, double contourSensitivity, off_t maxKeypoints, bool autoAlign, bool radialMask) {
	Settings::instance().show_gui = showGui;
	Settings::instance().number_of_frames = numberOfFrames;
	Settings::instance().match_tolerance = matchTolerance;
	Settings::instance().max_keypoints = maxKeypoints;
	Settings::instance().contour_sensitivity = contourSensitivity;
	Settings::instance().enable_auto_align = autoAlign;
	Settings::instance().enable_radial_mask = radialMask;
}

template<typename Twriter>
void morph(Mat &image1, Mat &image2, Twriter &output) {
	Mat morphed;

	std::vector<Point2f> srcPoints1, srcPoints2, morphedPoints, lastMorphedPoints;

	Mat corrected1, corrected2;
	Mat allContours1, allContours2;
	find_matches(image1, image2, corrected1, corrected2, srcPoints1, srcPoints2, allContours1, allContours2);
	if(srcPoints1.empty() || srcPoints2.empty()) {
		cerr << "No matches found. Inserting dups." << endl;
		for (size_t j = 0; j < Settings::instance().number_of_frames; ++j) {
			output.write(image1);
		}
		return;
	}
	prepare_matches(corrected1, corrected2, image1, image2, srcPoints1, srcPoints2);
	float step = 1.0 / Settings::instance().number_of_frames;
	double linear = 0;
	double shape = 0;
	double mask = 0;

	image1 = corrected1.clone();
	image2 = corrected2.clone();

	for (size_t j = 0; j < Settings::instance().number_of_frames; ++j) {
		if (!lastMorphedPoints.empty())
			srcPoints1 = lastMorphedPoints;
		morphedPoints.clear();

		linear = j * step;
		shape = ((1.0 / (1.0 - linear)) / Settings::instance().number_of_frames);
		mask = ((1.0 / (1.0 - linear)) / Settings::instance().number_of_frames);
		shape = 1 - pow(1 - shape, 3);
		//		mask = sin(pow(linear,3) * M_PI/2);
//		mask = pow(tan(tan(tan(shape * M_PI/4)* M_PI/4)* M_PI/4),4);
		if (shape > 1.0)
			shape = 1.0;

		morph_images(image1, image2, morphed, morphed.clone(), morphedPoints, srcPoints1, srcPoints2, allContours1, allContours2, shape, shape);
		image1 = morphed.clone();
		lastMorphedPoints = morphedPoints;
		output.write(morphed);

		show_image("morphed", morphed);
#ifndef _WASM
		if (Settings::instance().show_gui) {
			int key = waitKey(1);

			switch (key) {
			case ((int) ('q')):
				exit(0);
				break;
			}
		}
#endif
		std::cerr << int((j / double(Settings::instance().number_of_frames)) * 100.0) << "%\r";
	}
	morphed.release();
	srcPoints1.clear();
	srcPoints2.clear();
}
}
#endif
