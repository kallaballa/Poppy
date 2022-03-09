#ifndef SRC_POPPY_HPP_
#define SRC_POPPY_HPP_

#include "algo.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <opencv2/core/core.hpp>

namespace poppy {
void init(bool showGui, size_t numberOfFrames, double matchTolerance, double targetAngDiff, double targetLenDiff, double contourSensitivity, off_t maxKeypoints) {
	show_gui = showGui;
	number_of_frames = numberOfFrames;
	match_tolerance = matchTolerance;
	max_keypoints = maxKeypoints;
	target_ang_diff = targetAngDiff;
	target_len_diff = targetLenDiff;
	contour_sensitivity = contourSensitivity;
}

template<typename Twriter>
void morph(const Mat &src1, const Mat &src2, Twriter &output) {
	Mat image1 = src1.clone();
	Mat image2 = src2.clone();
	Mat orig1 = image1.clone();
	Mat orig2 = image2.clone();
	Mat morphed;

	std::vector<Point2f> srcPoints1, srcPoints2, morphedPoints, lastMorphedPoints;

	Mat allContours1, allContours2;
	find_matches(orig1, orig2, srcPoints1, srcPoints2, allContours1, allContours2);
	if(srcPoints1.empty() || srcPoints2.empty()) {
		for (size_t j = 0; j < number_of_frames; ++j) {
			output.write(image1);
		}
		return;
	}
	prepare_matches(orig1, orig2, image1, image2, srcPoints1, srcPoints2);

	float step = 1.0 / number_of_frames;
	double linear = 0;
	double shape = 0;
	double mask = 0;

	for (size_t j = 0; j < number_of_frames; ++j) {
		if (!lastMorphedPoints.empty())
			srcPoints1 = lastMorphedPoints;
		morphedPoints.clear();

		linear = j * step;
		shape = ((1.0 / (1.0 - linear)) / number_of_frames);
		mask = pow(shape, 1.1);
		if (shape > 1.0)
			shape = 1.0;

		morph_images(image1, orig2, morphed, morphed.clone(), morphedPoints, srcPoints1, srcPoints2, allContours1, allContours2, shape, mask);

		image1 = morphed.clone();
		lastMorphedPoints = morphedPoints;
		output.write(morphed);

		show_image("morphed", morphed);
		if (show_gui)
			waitKey(1);

		std::cerr << int((j / double(number_of_frames)) * 100.0) << "%\r";
	}
	morphed.release();
	srcPoints1.clear();
	srcPoints2.clear();
}
}
#endif
