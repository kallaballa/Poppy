#ifndef SRC_POPPY_HPP_
#define SRC_POPPY_HPP_

#include "util.hpp"
#include "algo.hpp"
#include "settings.hpp"
#include "face.hpp"
#include "matcher.hpp"
#include "extractor.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace poppy {

double ease(double x) {
	return 1.0 - pow(1.0 - x, 2.0);
}

double easeInOutSine(double x) {
	return -(cos(M_PI * x) - 1.0) / 2.0;
}

void init(bool showGui, size_t numberOfFrames, double matchTolerance, double contourSensitivity, bool autoAlign, bool radialMask, bool faceDetect, bool denoise, bool srcScaling, double frameRate, size_t pyramidLevels, string fourcc, bool enableWait, size_t faceNeighbors) {
	Settings::instance().show_gui = showGui;
	Settings::instance().enable_wait = enableWait;
	Settings::instance().number_of_frames = numberOfFrames;
	Settings::instance().frame_rate = frameRate;
	Settings::instance().match_tolerance = matchTolerance;
	Settings::instance().contour_sensitivity = contourSensitivity;
	Settings::instance().enable_auto_align = autoAlign;
	Settings::instance().enable_radial_mask = radialMask;
	Settings::instance().enable_denoise = denoise;
	Settings::instance().enable_src_scaling = srcScaling;
	Settings::instance().enable_face_detection = faceDetect;
	Settings::instance().pyramid_levels = pyramidLevels;
	Settings::instance().fourcc = fourcc;
	Settings::instance().face_neighbors = faceNeighbors;
}

template<typename Twriter>
void morph(const Mat &img1, const Mat &img2, Mat& corrected1, Mat& corrected2, double phase, bool distance, Twriter &output) {
	Mat morphed;
	std::vector<Point2f> srcPoints1, srcPoints2, morphedPoints, lastMorphedPoints;
	Features ft1;
	Features ft2;

	if(phase == 0) {
		cerr << "zero phase. inserting image 1" << endl;
		Mat cl1 = img1.clone();
		output.write(cl1);
		return;
	} else if (phase == 1 && !Settings::instance().enable_auto_align) {
		cerr << "full phase. inserting image 2" << endl;
		Mat cl2 = img2.clone();
		output.write(cl2);
		return;
	}

	if (Settings::instance().enable_face_detection) {
		ft1 = FaceDetector::instance().detect(img1);
		ft2 = FaceDetector::instance().detect(img2);
	}

	bool savedefd = Settings::instance().enable_face_detection;
	if(ft1.empty() || ft2.empty())
		Settings::instance().enable_face_detection = false;

	Extractor extractor(img1, img2);
	auto goodFeatures = extractor.prepareFeatures();
	Matcher matcher(img1, img2, ft1, ft2);
	matcher.find(corrected1, corrected2, srcPoints1, srcPoints2);
	Mat corr2Float;
	corrected2.convertTo(corr2Float, CV_32F, 1.0 / 255);
	Mat gabor2;
	gabor_filter(corr2Float, gabor2);
	show_image("gabor2", gabor2);

	if((srcPoints1.empty() || srcPoints2.empty()) && !distance) {
		cerr << "No matches found. Inserting linear blend." << endl;
		if(phase != -1) {
			//linear interpolation as fallback
			Mat blend = ((img2 * phase) + (img1 * (1.0 - phase)));
			output.write(blend);
			blend.release();
		} else {
			Mat blend;
			for (size_t j = 0; j < Settings::instance().number_of_frames; ++j) {
				blend = ((img2 * phase) + (img1 * (1.0 - phase)));
				output.write(blend);
			}
		}
		return;
	}

	if(!Settings::instance().enable_face_detection) {
		matcher.prepare(corrected1, corrected2, srcPoints1, srcPoints2);
	} else {
		add_corners(srcPoints1, srcPoints2, img1.size);
	}

	vector<Point2f> uniq1;
	clip_points(srcPoints1, img1.cols, img1.rows);
	check_points(srcPoints1, img1.cols, img1.rows);
	make_uniq(srcPoints1, uniq1);
	check_uniq(uniq1);

	vector<Point2f> uniq2;
	clip_points(srcPoints2, img1.cols, img1.rows);
	check_points(srcPoints2, img1.cols, img1.rows);
	make_uniq(srcPoints2, uniq2);
	check_uniq(uniq2);

	if (uniq1.size() > uniq2.size())
		uniq1.resize(uniq2.size());
	else
		uniq2.resize(uniq1.size());

	cerr << "inital morph distance: " << morph_distance2(uniq1, uniq2, img1.cols, img1.rows) << endl;

	if(distance) {
		exit(0);
	}

//	if(md == 0.0) {
//		cerr << "Zero morph distance. Inserting one frame..." << endl;
//		output.write(img2);
//		Settings::instance().enable_face_detection = savedefd;
//		return;
//	}

	double linear = 0;
	double shape = 0;
	double progress = 0;
	double color = 0;

	for (size_t j = 0; j < Settings::instance().number_of_frames; ++j) {
		if (!lastMorphedPoints.empty())
			srcPoints1 = lastMorphedPoints;
		morphedPoints.clear();
		linear = (j / double(Settings::instance().number_of_frames));

//		if(Settings::instance().number_of_frames == 1)
//			phase = 0.5;

		if(phase >= 1.0)
			progress = 1;
		if(phase < 1.0 && phase >= 0)
			progress = 1.0 / Settings::instance().number_of_frames;
		else if(linear == 0)
			progress = 0;
		else if(linear == 1)
			progress = 1;
		else
			progress = (1.0 / (1.0 - linear)) / Settings::instance().number_of_frames;

		if(phase < 1.0 && phase >= 0)
			shape = progress * phase;
		else
			shape = progress;

//		shape = log(ease(progress) * 2.0 + 1.0) / log(3.0);

		if(shape > 1)
			shape = 1;
		color = shape;
//		color = log(ease(progress) * 2.0 + 1.0) / log(3.0);

		if(color > 1)
			color = 1;

//		cerr << "morph distance: " << scientific << morph_distance2(srcPoints1, srcPoints2, img1.cols, img1.rows) << fixed << endl;

//		cerr << linear << "\t| " << progress << "\t| " << shape << "\t| " << color << endl;
		morph_images(img1, img2, corrected1, corrected2, gabor2, goodFeatures.first, goodFeatures.second, morphed, morphed.clone(), morphedPoints, srcPoints1, srcPoints2, shape, color, linear);

		corrected1 = morphed.clone();
		lastMorphedPoints = morphedPoints;
		output.write(morphed);
		show_image("morphed", morphed);
		wait_key();

	#ifndef _WASM
		if (Settings::instance().show_gui) {
			int key = waitKey(1);

			switch (key) {
			case ((int) ('q')):
				exit(0);
				break;
			}
		}

		if(phase >= 0)
			break;
#endif
		std::cerr << int((linear) * 100.0) << "%";
#ifdef _WASM
		std::cerr << std::endl;
#else
		std::cerr << '\r';
#endif
	}
	Settings::instance().enable_face_detection = savedefd;

	morphed.release();
	srcPoints1.clear();
	srcPoints2.clear();
}
}
#endif
