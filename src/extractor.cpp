#include "extractor.hpp"

#include "settings.hpp"
#include "util.hpp"
#include "draw.hpp"
#include "experiments.hpp"

#include <iostream>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

using std::cerr;
using std::cout;
using std::endl;

namespace poppy {

Extractor::Extractor(const Mat& img1, const Mat& img2) : img1_(img1), img2_(img2) {
	cvtColor(img1_, grey1_, COLOR_RGB2GRAY);
	cvtColor(img2_, grey2_, COLOR_RGB2GRAY);
}

Extractor::~Extractor() {
}

pair<Mat, Mat> Extractor::prepareFeatures() {
	foreground(goodFeatures1_, goodFeatures2_);
	return { goodFeatures1_, goodFeatures2_};
}

pair<vector<Point2f>, vector<Point2f>> Extractor::keypointsRaw() {
	cerr << "extract keypoints raw..." << endl;
	Ptr<ORB> detector = ORB::create(Settings::instance().max_keypoints);
	vector<KeyPoint> keypoints1, keypoints2;
	Mat trip1, trip2;

	triple_channel(goodFeatures1_, trip1);
	triple_channel(goodFeatures2_, trip2);
	trip1.convertTo(trip1, CV_32F, 1.0/255.0);
	trip2.convertTo(trip2, CV_32F, 1.0/255.0);

	Mat us1 = unsharp_mask(trip1, 2, 3, 0.1);
	Mat us2 = unsharp_mask(trip2, 2, 3, 0.1);
	cvtColor(us1, us1, COLOR_BGR2GRAY);
	cvtColor(us2, us2, COLOR_BGR2GRAY);

	Mat radial = draw_radial_gradiant2(us1.cols, us1.rows);
	Mat g1, g2;

	gabor_filter(us1,g1, 16, 31, 5, 2, 0.04,CV_PI/4);
	gabor_filter(us2,g2, 16, 31, 5, 2, 0.04,CV_PI/4);

	multiply(g1, us1, g1);
	multiply(g2, us2, g2);
	multiply(g1, radial, g1);
	multiply(g2, radial, g2);

	g1.convertTo(g1, CV_8U, 255.0);
	g2.convertTo(g2, CV_8U, 255.0);
	equalizeHist(g1,g1);
	equalizeHist(g2,g2);
	detector->detect(g1, keypoints1);
	detector->detect(g2, keypoints2);

	cerr << "unfiltered keypoints: " << std::min(keypoints1.size(), keypoints2.size()) << endl;

	vector<Point2f> points1, points2;
	for (auto pt1 : keypoints1)
		points1.push_back(pt1.pt);

	for (auto pt2 : keypoints2)
		points2.push_back(pt2.pt);

	if (points1.size() > points2.size())
		points1.resize(points2.size());
	else
		points2.resize(points1.size());

	cerr << "keypoints extracted: " << points1.size() << endl;


	triple_channel(g1, g1);
	triple_channel(g2, g2);
	Mat bgr1 = g1.clone();
	Mat bgr2 = g2.clone();
	plot(bgr1, points1, Scalar(0,0,255), 1);
	plot(bgr2, points2, Scalar(0,255,0), 1);

	show_image("pts1", bgr1);
	show_image("pts2", bgr2);

	return {points1,points2};
}

void Extractor::foregroundMask(const Mat &grey, Mat &fgMask, const size_t& iterations) {
	// create a foreground mask by blurring the image over again and tracking the flow of pixels.
	fgMask = Mat::zeros(grey.rows, grey.cols, grey.type());
	Mat last = grey.clone();
	Mat fgMaskBlur;
	Mat med, flow;

	//optical flow tracking works as well but is much slower
	auto pBackSub1 = createBackgroundSubtractorMOG2();
	pBackSub1->apply(grey, flow);
	fgMask += (flow * (1.0 / (iterations / 2.0)));

	for (size_t i = 0; i < 12; ++i) {
		medianBlur(last, med, i * 8 + 1);
		pBackSub1->apply(med, flow);
		fgMask += (flow * (1.0 / (iterations / 2.0)));
		//FIXME do we really need to blur here?
		GaussianBlur(fgMask, fgMaskBlur, { 23, 23 }, 1);
		fgMask = fgMaskBlur.clone();
		last = med.clone();
		med.release();
		flow.release();
		fgMaskBlur.release();
	}
	last.release();
}

void Extractor::foreground(Mat &foreground1, Mat &foreground2) {
	cerr << "extract foreground..." << endl;
	Mat grey1, grey2;
	Mat fgMask1;
	Mat fgMask2;

	cvtColor(img1_, grey1, COLOR_BGR2GRAY);
	cvtColor(img2_, grey2, COLOR_BGR2GRAY);

	//extract areas of interest (aka. foreground)
	foregroundMask(grey1, fgMask1);
	foregroundMask(grey2, fgMask2);
	show_image("fgm1", fgMask1);
	show_image("fgm2", fgMask2);

	Mat radialMaskFloat;
	if (Settings::instance().enable_radial_mask) {
		//create a radial mask to bias the contrast towards the center
		Mat radial = Mat::ones(grey1.rows, grey1.cols, CV_32F);
		draw_radial_gradiant(radial);
		radial.convertTo(radialMaskFloat, CV_32F, 1.0 / 255.0);
		radial.release();
	}
	//convert the images and masks to floating point for the subsequent multiplication
	Mat grey1Float, grey2Float, fgMask1Float, fgMask2Float;
	grey1.convertTo(grey1Float, CV_32F, 1.0 / 255.0);
	grey2.convertTo(grey2Float, CV_32F, 1.0 / 255.0);
	fgMask1.convertTo(fgMask1Float, CV_32F, 1.0 / 255.0);
	fgMask2.convertTo(fgMask2Float, CV_32F, 1.0 / 255.0);

	//multiply the fg mask with the radial mask to emphasize features in the center of the image
	Mat finalMask1Float, finalMask2Float;
	if (Settings::instance().enable_radial_mask) {
		multiply(fgMask1Float, radialMaskFloat, finalMask1Float);
		multiply(fgMask2Float, radialMaskFloat, finalMask2Float);
	} else {
		fgMask1Float.copyTo(finalMask1Float);
		fgMask2Float.copyTo(finalMask2Float);
	}

	int logBase = 20;
	Mat logMask(finalMask1Float.size(),CV_32F);
	logMask = Scalar::all(logBase);
	log(logMask, logMask);
	log(finalMask1Float * (logBase - 1.0) + 1.0, finalMask1Float);
	log(finalMask2Float * (logBase - 1.0) + 1.0, finalMask2Float);
	divide(finalMask1Float, logMask, finalMask1Float);
	divide(finalMask2Float, logMask, finalMask2Float);

	show_image("mk1", finalMask1Float);
	show_image("mk2", finalMask2Float);

	Mat masked1, masked2;
	multiply(grey1Float, finalMask1Float, masked1);
	multiply(grey2Float, finalMask2Float, masked2);

	masked1.convertTo(masked1, CV_8U, 255.0);
	masked2.convertTo(masked2, CV_8U, 255.0);

	equalizeHist(masked1, foreground1);
	equalizeHist(masked2, foreground2);
	masked1.release();
	masked2.release();

	show_image("fg1", foreground1);
	show_image("fg2", foreground2);
}

} /* namespace poppy */
