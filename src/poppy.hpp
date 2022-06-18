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

double morph_distance(double width, double height, vector<Point2f> srcPoints1, vector<Point2f> srcPoints2) {
	assert(srcPoints1.size() == srcPoints2.size());
	double totalDistance = 0;
	for(size_t i = 0; i < srcPoints1.size(); ++i) {
		Point2f v = srcPoints2[i] - srcPoints1[i];
		totalDistance += hypot(v.x, v.y);
	}

	return (totalDistance / srcPoints1.size()) / hypot(width, height);
}

void init(bool showGui, size_t numberOfFrames, double matchTolerance, double contourSensitivity, off_t maxKeypoints, bool autoAlign, bool radialMask, bool faceDetect, bool denoise, bool srcScaling, double frameRate) {
	Settings::instance().show_gui = showGui;
	Settings::instance().number_of_frames = numberOfFrames;
	Settings::instance().frame_rate = frameRate;
	Settings::instance().match_tolerance = matchTolerance;
	Settings::instance().max_keypoints = maxKeypoints;
	Settings::instance().contour_sensitivity = contourSensitivity;
	Settings::instance().enable_auto_align = autoAlign;
	Settings::instance().enable_radial_mask = radialMask;
	Settings::instance().enable_denoise = denoise;
	Settings::instance().enable_src_scaling = srcScaling;
	Settings::instance().enable_face_detection = faceDetect;
}

template<typename Twriter>
void morph(Mat &image1, Mat &image2, double phase, bool distance, Twriter &output) {
	Mat morphed;

	std::vector<Point2f> srcPoints1, srcPoints2, morphedPoints, lastMorphedPoints;

	Mat corrected1, corrected2;
	Mat allContours1, allContours2;
	std::cerr << "usage2: " << (uintptr_t) sbrk(0) << std::endl;
	find_matches(image1, image2, corrected1, corrected2, srcPoints1, srcPoints2, allContours1, allContours2);
	std::cerr << "usage2a: " << (uintptr_t) sbrk(0) << std::endl;

	if((srcPoints1.empty() || srcPoints2.empty()) && !distance) {
		cerr << "No matches found. Inserting dups." << endl;
		if(phase != -1) {
			output.write(image1);
		} else {
			for (size_t j = 0; j < Settings::instance().number_of_frames; ++j) {
				output.write(image1);
			}
		}
		return;
	}
	std::cerr << "usage2b: " << (uintptr_t) sbrk(0) << std::endl;
	if(!Settings::instance().enable_face_detection) {
		prepare_matches(corrected1, corrected2, image1, image2, srcPoints1, srcPoints2);
	} else {
		add_corners(srcPoints1, srcPoints2, image1.size);
	}
	if(distance) {
		std::cerr << morph_distance(double(image1.size().width), double(image1.size().height), srcPoints1, srcPoints2) << std::endl;
		exit(0);
	}

	float step = 1.0 / Settings::instance().number_of_frames;
	double linear = 0;

	image1 = corrected1.clone();
	image2 = corrected2.clone();

	for (size_t j = 0; j < Settings::instance().number_of_frames; ++j) {
		if (!lastMorphedPoints.empty())
			srcPoints1 = lastMorphedPoints;
		morphedPoints.clear();

		if(phase != -1)
			linear = j * step * 0.6 * phase;
		else
			linear = j * step * 0.6;

		if(linear > 1.0)
			linear = 1.0;

		morph_images(image1, image2, morphed, morphed.clone(), morphedPoints, srcPoints1, srcPoints2, allContours1, allContours2, linear, linear);
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
		std::cerr << int((j / double(Settings::instance().number_of_frames - 1)) * 100.0) << "%";
#ifdef _WASM
		std::cerr << std::endl;
#else
		std::cerr << '\r';
#endif
	}
	morphed.release();
	srcPoints1.clear();
	srcPoints2.clear();
}
}
#endif
