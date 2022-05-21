#include "face.hpp"
#ifndef _NO_FACE_DETECT

#include <sstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

namespace poppy {
namespace d = dlib;
using namespace cv;
using namespace std;

FaceDetector::FaceDetector() {
	d::deserialize("/home/elchaschab/devel/Poppy/assets/shape_predictor_68_face_landmarks.dat") >> sp_;
	detector_ = d::get_frontal_face_detector();
}

Features FaceDetector::detect(const cv::Mat &frame) {
	Features features;

	Mat img = frame.clone();
	d::array2d<d::bgr_pixel> dlibFrame;
	d::assign_image(dlibFrame, dlib::cv_image<d::bgr_pixel>(img));
	size_t orig_rows = dlibFrame.nr();
	size_t orig_columns = dlibFrame.nc();
	d::pyramid_up(dlibFrame);
	size_t scaled_rows = dlibFrame.nr();
	size_t scaled_columns = dlibFrame.nc();

	double rf = (double) scaled_rows / (double) orig_rows;
	double cf = (double) scaled_columns / (double) orig_columns;

	std::vector<dlib::rectangle> dets = detector_(dlibFrame);
	cout << "Number of faces detected: " << dets.size() << endl;
	if (dets.empty())
		return {};
	std::vector<d::full_object_detection> shapes;

	d::full_object_detection shape = sp_(dlibFrame, dets[0]);
	Point2f nose_bottom(0, 0);
	Point2f lips_top(0, std::numeric_limits<float>().max());
	cout << "number of parts: " << shape.num_parts() << endl;

	// Around Chin. Ear to Ear
	for (unsigned long i = 1; i <= 16; ++i)
		features.chin_.push_back(Point2f(shape.part(i).x() / rf, shape.part(i).y() / cf));

	// Line on top of nose

	for (unsigned long i = 28; i <= 30; ++i)
		features.top_nose_.push_back(Point2f(shape.part(i).x() / rf, shape.part(i).y() / cf));

	// left eyebrow
	for (unsigned long i = 18; i <= 21; ++i)
		features.left_eyebrow_.push_back(Point2f(shape.part(i).x() / rf, shape.part(i).y() / cf));

	// Right eyebrow
	for (unsigned long i = 23; i <= 26; ++i)
		features.right_eyebrow_.push_back(Point2f(shape.part(i).x() / rf, shape.part(i).y() / cf));

	// Bottom part of the nose
	for (unsigned long i = 31; i <= 35; ++i)
		features.bottom_nose_.push_back(Point2f(shape.part(i).x() / rf, shape.part(i).y() / cf));

	// Left eye
	for (unsigned long i = 37; i <= 41; ++i)
		features.left_eye_.push_back(Point2f(shape.part(i).x() / rf, shape.part(i).y() / cf));

	// Right eye
	for (unsigned long i = 43; i <= 47; ++i)
		features.right_eye_.push_back(Point2f(shape.part(i).x() / rf, shape.part(i).y() / cf));

	// Lips outer part
	for (unsigned long i = 49; i <= 59; ++i)
		features.outer_lips_.push_back(Point2f(shape.part(i).x() / rf, shape.part(i).y() / cf));

	// Lips inside part
	for (unsigned long i = 61; i <= 67; ++i)
		features.inside_lips_.push_back(Point2f(shape.part(i).x() / rf, shape.part(i).y() / cf));

	return features;
}
} /* namespace poppy */
#endif
