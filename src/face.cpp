#include "face.hpp"
#include "util.hpp"

#include <sstream>
#include <vector>
#include <string>
#ifndef _NO_FACE_DETECT
#include <opencv2/face/facemark.hpp>
#endif

namespace poppy {
using namespace cv;
using namespace std;


FaceDetector::FaceDetector(std::string cascade_model, double scale) : cfg(cascade_model, scale) {
#ifndef _NO_FACE_DETECT
	FacemarkLBF::Params params;
	params.model_filename = "assets/lbfmodel.yaml";
	facemark->loadModel(params.model_filename);
#endif
}

Features FaceDetector::detect(const cv::Mat &frame) {
	Features features;
#ifndef _NO_FACE_DETECT
	Mat img = frame.clone();

	vector<Rect> faces;
	resize(img,img,Size(460,460),0,0,INTER_LINEAR_EXACT);
	Mat gray;
	if(img.channels()>1){
	    cvtColor(img,gray,COLOR_BGR2GRAY);
	}
	else{
	    gray = img.clone();
	}
	equalizeHist( gray, gray );
	cfg.face_detector.detectMultiScale( gray, faces, 1.1, 3,0, Size(30, 30) );

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
		features.chin_.push_back(shapes[0][i]);

	// left eyebrow
	for (;i <= 21; ++i)
		features.left_eyebrow_.push_back(shapes[0][i]);

	// Right eyebrow
	for (; i <= 26; ++i)
		features.right_eyebrow_.push_back(shapes[0][i]);

	// Line on top of nose
	for (; i <= 30; ++i)
		features.top_nose_.push_back(shapes[0][i]);


	// Bottom part of the nose
	for (; i <= 35; ++i)
		features.bottom_nose_.push_back(shapes[0][i]);

	// Left eye
	for (unsigned long i = 37; i <= 41; ++i)
		features.left_eye_.push_back(shapes[0][i]);

	// Right eye
	for (unsigned long i = 43; i <= 47; ++i)
		features.right_eye_.push_back(shapes[0][i]);

	// Lips outer part
	for (unsigned long i = 49; i <= 59; ++i)
		features.outer_lips_.push_back(shapes[0][i]);

	// Lips inside part
	for (unsigned long i = 61; i <= 67; ++i)
		features.inside_lips_.push_back(shapes[0][i]);
#endif
	return features;
}
} /* namespace poppy */
