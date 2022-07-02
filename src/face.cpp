#include "face.hpp"
#include "util.hpp"

#include <sstream>
#include <vector>
#include <string>
#ifndef _NO_FACE_DETECT
#include <opencv2/face/facemark.hpp>
#endif


namespace poppy {

FaceDetector* FaceDetector::instance_ = nullptr;
FaceDetector::FaceDetector(double scale) : cfg(scale) {
#ifndef _NO_FACE_DETECT
#ifndef _WASM
	face_detector.load("src/assets/lbpcascade_frontalface.xml");
	if(face_detector.empty())
		face_detector.load("../src/assets/lbpcascade_frontalface.xml");
#else
	face_detector.load("assets/lbpcascade_frontalface.xml");
#endif

	facemark = FacemarkLBF::create();
#ifndef _WASM
	try {
		facemark->loadModel("src/assets/lbfmodel.yaml");
	}	catch(...) {
		facemark->loadModel("../src/assets/lbfmodel.yaml");
	}
#else
	facemark->loadModel("assets/lbfmodel.yaml");
#endif
#endif
}

Features FaceDetector::detect(const cv::Mat &frame) {
	Features features;
#ifndef _NO_FACE_DETECT
	if(frame.empty())
		return features;

	Mat img = frame.clone();

	vector<Rect> faces;
	double scale = 1;
//	double scale = 0.25;
//	double width = img.cols * scale;
//	double height = img.rows * scale;
//	resize(img,img,Size(width,height),0,0,INTER_LINEAR_EXACT);
	Mat gray;
	cvtColor(img,gray,COLOR_BGR2GRAY);
	equalizeHist( gray, gray );
	face_detector.detectMultiScale( gray, faces, 1.1, 4, 0, Size(30, 30) );

	cerr << "Number of faces detected: " << faces.size() << endl;
	if (faces.empty())
		return {};

	vector< vector<Point2f> > shapes;
	if(!facemark->fit(img,faces,shapes)){
		cerr << "No facemarks detected." << endl;
		return {};
	}

	Point2f nose_bottom(0, 0);
	Point2f lips_top(0, std::numeric_limits<float>().max());
	unsigned long i = 0;
	// Around Chin. Ear to Ear
	for (i = 0; i <= 16; ++i)
		features.chin_.push_back(shapes[0][i] / scale);

	// left eyebrow
	for (;i <= 21; ++i)
		features.left_eyebrow_.push_back(shapes[0][i] / scale);

	// Right eyebrow
	for (; i <= 26; ++i)
		features.right_eyebrow_.push_back(shapes[0][i] / scale);

	// Line on top of nose
	for (; i <= 30; ++i)
		features.top_nose_.push_back(shapes[0][i] / scale);


	// Bottom part of the nose
	for (; i <= 35; ++i)
		features.bottom_nose_.push_back(shapes[0][i] / scale);

	// Left eye
	for (unsigned long i = 37; i <= 41; ++i)
		features.left_eye_.push_back(shapes[0][i] / scale);

	// Right eye
	for (unsigned long i = 43; i <= 47; ++i)
		features.right_eye_.push_back(shapes[0][i] / scale);

	// Lips outer part
	for (unsigned long i = 49; i <= 59; ++i)
		features.outer_lips_.push_back(shapes[0][i] / scale);

	// Lips inside part
	for (unsigned long i = 61; i <= 67; ++i)
		features.inside_lips_.push_back(shapes[0][i] / scale);
#endif
	return features;
}
} /* namespace poppy */
