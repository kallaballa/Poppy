#ifndef SRC_POPPY_HPP_
#define SRC_POPPY_HPP_

#include "util.hpp"
#include "algo.hpp"
#include "settings.hpp"
#include "face.hpp"
#include "matcher.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace poppy {

double ease(double x) {
	return 1.0 - pow(1.0 - x, 4.0);
}

double easeInOutSine(double x) {
	return -(cos(M_PI * x) - 1.0) / 2.0;
}

void init(bool showGui, size_t numberOfFrames, double matchTolerance, double contourSensitivity, off_t maxKeypoints, bool autoAlign, bool radialMask, bool faceDetect, bool denoise, bool srcScaling, double frameRate, size_t pyramidLevels, string fourcc) {
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
	Settings::instance().pyramid_levels = pyramidLevels;
	Settings::instance().fourcc = fourcc;
}

template<typename Twriter>
void morph(Mat &image1, Mat &image2, double phase, bool distance, Twriter &output) {
	bool savedefd = Settings::instance().enable_face_detection;
	Mat morphed;
	Matcher matcher;
	std::vector<Point2f> srcPoints1, srcPoints2, morphedPoints, lastMorphedPoints;

	Mat corrected1, corrected2;
	Mat contourMap1, contourMap2;
	Mat edges1, edges2;
	Features ft1;
	Features ft2;

	if (Settings::instance().enable_face_detection) {
		ft1 = FaceDetector::instance().detect(image1);
		ft2 = FaceDetector::instance().detect(image2);
	}

	if(ft1.empty() || ft2.empty())
		Settings::instance().enable_face_detection = false;

	matcher.find(image1, image2, ft1, ft2, corrected1, corrected2, srcPoints1, srcPoints2, contourMap1, contourMap2, edges1, edges2);

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
	if(!Settings::instance().enable_face_detection) {
		matcher.prepare(corrected1, corrected2, image1, image2, srcPoints1, srcPoints2);
	} else {
		add_corners(srcPoints1, srcPoints2, image1.size);
	}
	if(distance) {
		vector<Point2f> uniq1;
		clip_points(srcPoints1, image1.cols, image1.rows);
		check_points(srcPoints1, image1.cols, image1.rows);
		make_uniq(srcPoints1, uniq1);
		check_uniq(uniq1);

		vector<Point2f> uniq2;
		clip_points(srcPoints2, image1.cols, image1.rows);
		check_points(srcPoints2, image1.cols, image1.rows);
		make_uniq(srcPoints2, uniq2);
		check_uniq(uniq2);

		cerr << morph_distance(srcPoints1, srcPoints2, image1.cols, image1.rows) << endl;
		exit(0);
	}

	float step = 1.0 / Settings::instance().number_of_frames;
	double linear = 0;
	double shape = 0;
	double progress = 0;
	double color = 0;
	image1 = corrected1.clone();
	image2 = corrected2.clone();

	for (size_t j = 0; j < Settings::instance().number_of_frames; ++j) {
		if (!lastMorphedPoints.empty())
			srcPoints1 = lastMorphedPoints;
		morphedPoints.clear();

		if(phase != -1)
			linear = j * step * phase;
		else
			linear = j * step;

		if(linear > 1.0)
			linear = 1.0;

		progress = (1.0 / (1.0 - linear)) / Settings::instance().number_of_frames;
		shape = progress;
		color = shape;
		morph_images(image1, image2, contourMap1, contourMap2, edges1, edges2, morphed, morphed.clone(), morphedPoints, srcPoints1, srcPoints2, shape, color);
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
	Settings::instance().enable_face_detection = savedefd;
}
}
#endif
