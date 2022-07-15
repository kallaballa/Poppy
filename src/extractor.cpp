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
	Mat dst1, dst2;
	double detail1 = dft_detail(goodFeatures1_, dst1) / (goodFeatures1_.cols * goodFeatures1_.rows);
	double detail2 = dft_detail(goodFeatures2_, dst2) / (goodFeatures2_.cols * goodFeatures2_.rows);
	Ptr<ORB> detector = ORB::create(1.0 / detail1 * 100 + 1.0 / detail2 * 100);
	vector<KeyPoint> keypoints1, keypoints2;
	Mat trip1, trip2;
	triple_channel(goodFeatures1_, trip1);
	triple_channel(goodFeatures2_, trip2);
    cvtColor(trip1, trip1, COLOR_BGR2GRAY);
    cvtColor(trip2, trip2, COLOR_BGR2GRAY);
	show_image("ur1", trip1);
	show_image("ur2", trip2);

//	Mat descriptors1, descriptors2;
	detector->detect(trip1, keypoints1);
	detector->detect(trip2, keypoints2);

//	detector->compute(grey1, keypoints1, descriptors1);
//	detector->compute(grey2, keypoints2, descriptors2);

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

	return {points1,points2};
}

pair<vector<Point2f>, vector<Point2f>> Extractor::keypointsFlann() {
	cerr << "extract keypoints flann..." << endl;
	Mat dst1, dst2;

	double detail1 = dft_detail(goodFeatures1_, dst1) / (goodFeatures1_.cols * goodFeatures1_.rows);
	double detail2 = dft_detail(goodFeatures2_, dst2) / (goodFeatures2_.cols * goodFeatures2_.rows);
	Ptr<ORB> detector = ORB::create((1.0 / detail1) * 1000 + (1.0 / detail2) * 1000);

	vector<KeyPoint> keypoints1, keypoints2;
	Mat trip1, trip2;
	triple_channel(goodFeatures1_, trip1);
	triple_channel(goodFeatures2_, trip2);
    cvtColor(trip1, trip1, COLOR_BGR2GRAY);
    cvtColor(trip2, trip2, COLOR_BGR2GRAY);

	show_image("uf1", trip1);
	show_image("uf2", trip2);

	Mat descriptors1, descriptors2;
	detector->detect(trip1, keypoints1);
	detector->detect(trip1, keypoints2);

	detector->compute(trip1, keypoints1, descriptors1);
	detector->compute(trip1, keypoints2, descriptors2);


	cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::LshIndexParams(6, 12, 1);
	FlannBasedMatcher matcher(indexParams);
	std::vector<std::vector<cv::DMatch>> matches;
	matcher.knnMatch(descriptors1, descriptors2, matches, 2);

	std::vector<Point2f> points1;
	std::vector<Point2f> points2;

//	ratioTest(matches);

	for(auto& v : matches) {
		for(auto& dm : v) {
			points1.push_back(keypoints1[dm.queryIdx].pt);
			points2.push_back(keypoints2[dm.trainIdx].pt);
		}
	}

	cerr << "keypoints extracted: " << points1.size() << endl;

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

void Extractor::contours(Mat &contourMap1, Mat &contourMap2, vector<Mat>& contourLayers1, vector<Mat>& contourLayers2) {
	cerr << "extract contours..." << endl;
	Mat grey1 = grey1_.clone();
	Mat grey2 = grey2_.clone();
	Mat blur1, blur2;

	vector<vector<vector<Point2f>>> collected1;
	vector<vector<vector<Point2f>>> collected2;

	equalizeHist(grey1, grey1);
	equalizeHist(grey2, grey2);
	GaussianBlur(grey1, blur1, {9,9}, 1);
	GaussianBlur(grey2, blur2, {9,9}, 1);

	double t1 = 0;
	double t2 = 255;
	Mat thresh1, thresh2;
	vector<Vec4i> hierarchy1;
	size_t numContours = 0;

	t1 = 0;
	t2 = 0;
	cerr << "thresholding 1" << endl;
	for (off_t i = 0; i < 16; ++i) {
		t1 = max(0, min(255, (int) round((i * 16 * Settings::instance().contour_sensitivity))));
		t2 = max(0, min(255, (int) round(((i + 1) * 16 * Settings::instance().contour_sensitivity))));
		cerr << i + 1 << "/" << 16 << '\r';

		threshold(blur1, thresh1, t1, t2, 0);

		vector<vector<Point>> contours1;
		findContours(thresh1, contours1, hierarchy1, RETR_TREE, CHAIN_APPROX_TC89_KCOS);
		std::vector<std::vector<Point2f>> tmp = convertContourTo2f(contours1);

		assert(!tmp.empty());
		collected1.push_back(tmp);
		numContours += tmp.size();
	}
	cerr << endl;

	Mat cmap1, cmap2;
	cerr << "draw map 1: " << numContours << endl;
	numContours = 0;

	draw_contour_map(cmap1, contourLayers1, collected1, hierarchy1, grey1.cols, grey1.rows, grey1.type());

	cerr << "thresholding 2" << endl;
	vector<Vec4i> hierarchy2;
	for (off_t j = 0; j < 16; ++j) {
		t1 = min(255, (int) round((j * 16.0 * Settings::instance().contour_sensitivity)));
		t2 = min(255, (int) round(((j + 1) * 16.0 * Settings::instance().contour_sensitivity)));
		cerr << j + 1 << "/" << 16 << '\r';

		threshold(blur2, thresh2, t1, t2, 0);

		vector<vector<Point>> contours2;
		findContours(thresh2, contours2, hierarchy2, RETR_TREE, CHAIN_APPROX_TC89_KCOS);
		std::vector<std::vector<Point2f>> tmp = convertContourTo2f(contours2);

		assert(!tmp.empty());
		collected2.push_back(tmp);
		numContours += tmp.size();
	}
	cerr << endl;

	cerr << "draw map 2: " << numContours << endl;
	draw_contour_map(cmap2, contourLayers2, collected2, hierarchy2, grey2.cols, grey2.rows, grey2.type());
	contourMap1_ = cmap1.clone();
	contourMap2_ = cmap2.clone();
	contourMap1 = cmap1.clone();
	contourMap2 = cmap2.clone();
	show_image("cmap1", cmap1);
	show_image("cmap2", cmap2);
	assert(contourLayers1.size() == contourLayers2.size());
}

void Extractor::reduceBackground(const Mat& img1, const Mat& img2, Mat& reduced1, Mat& reduced2) {
	Mat finalMask1Float, finalMask2Float;
	{
		Mat hue;
		int bins = 6;
		Mat hsv;
		cvtColor(img1, hsv, COLOR_BGR2HSV);
		hue.create(hsv.size(), hsv.depth());
		int ch[] = { 0, 0 };
		mixChannels(&hsv, 1, &hue, 1, ch, 1);
		int histSize = MAX(bins, 2);
		float hue_range[] = { 0, 180 };
		const float *ranges[] = { hue_range };
		Mat hist;
		calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);
		normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
		Mat backproj;
		calcBackProject(&hue, 1, 0, hist, backproj, ranges, 1, true);
		Mat filtered;
		Mat filteredFloat;
		inRange(backproj, { 254 }, { 255 }, filtered);
		backproj.convertTo(finalMask1Float, CV_32F, 1 / 255.0);
		finalMask1Float = 1.0 - finalMask1Float;
	}
	{
		Mat hue;
		int bins = 6;
		Mat hsv;
		cvtColor(img2, hsv, COLOR_BGR2HSV);
		hue.create(hsv.size(), hsv.depth());
		int ch[] = { 0, 0 };
		mixChannels(&hsv, 1, &hue, 1, ch, 1);
		int histSize = MAX(bins, 2);
		float hue_range[] = { 0, 180 };
		const float *ranges[] = { hue_range };
		Mat hist;
		calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);
		normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
		Mat backproj;
		calcBackProject(&hue, 1, 0, hist, backproj, ranges, 1, true);
		Mat filtered;
		Mat filteredFloat;
		inRange(backproj, { 254 }, { 255 }, filtered);
		backproj.convertTo(finalMask2Float, CV_32F, 1 / 255.0);
		finalMask2Float = 1.0 - finalMask2Float;
	}


	if(countNonZero(finalMask1Float) == 0) {
		finalMask1Float = Scalar::all(1);
	}

	if(countNonZero(finalMask2Float) == 0) {
		finalMask2Float = Scalar::all(1);
	}

	show_image("fin1",finalMask1Float);
	show_image("fin2",finalMask2Float);

	Mat img1Float, img2Float;
	img1.convertTo(img1Float, CV_32F, 1 / 255.0);
	img2.convertTo(img2Float, CV_32F, 1 / 255.0);

	Mat finalMask1FloatC3 = Mat::zeros(img1.rows, img1.cols, CV_32FC3);
	Mat finalMask2FloatC3 = Mat::zeros(img2.rows, img2.cols, CV_32FC3);
	Mat dilated1, dilated2;
	dilate(finalMask1Float, dilated1, getStructuringElement(MORPH_ELLIPSE, Size(11, 11)));
	dilate(finalMask2Float, dilated2, getStructuringElement(MORPH_ELLIPSE, Size(11, 11)));
	GaussianBlur(dilated1, finalMask1Float, { 127, 127 }, 12);
	GaussianBlur(dilated2, finalMask2Float, { 127, 127 }, 12);

	vector<Mat> planes1, planes2;
	for (int i = 0; i < 3; i++) {
		planes1.push_back(finalMask1Float);
		planes2.push_back(finalMask2Float);
	}
	merge(planes1, finalMask1FloatC3);
	merge(planes2, finalMask2FloatC3);

	Mat blurred1Float, blurred2Float;
	Mat maskedBlur1Float, maskedBlur2Float;
	Mat masked1Float, masked2Float;
	Mat invFinalMask1 = Scalar(1.0, 1.0, 1.0) - finalMask1FloatC3;
	Mat invFinalMask2 = Scalar(1.0, 1.0, 1.0) - finalMask2FloatC3;
	Mat combined1, combined2;
	GaussianBlur(img1Float, blurred1Float, { 63, 63 }, 5);
	GaussianBlur(img2Float, blurred2Float, { 63, 63 }, 5);
	multiply(blurred1Float, invFinalMask1, maskedBlur1Float);
	multiply(blurred2Float, invFinalMask2, maskedBlur2Float);
	multiply(img1Float, finalMask1FloatC3, masked1Float);
	multiply(img2Float, finalMask2FloatC3, masked2Float);
	add(masked1Float, maskedBlur1Float, combined1);
	add(masked2Float, maskedBlur2Float, combined2);

//	poppy::show_image("fmf1", finalMask1FloatC3);
//	poppy::show_image("fmf2", finalMask2FloatC3);
//	poppy::show_image("iv1", invFinalMask1);
//	poppy::show_image("iv2", invFinalMask2);
//	poppy::show_image("b1", blurred1Float);
//	poppy::show_image("b2", blurred2Float);
//	poppy::show_image("mb1", maskedBlur1Float);
//	poppy::show_image("mb2", maskedBlur2Float);
//	poppy::show_image("ig1", img1Float);
//	poppy::show_image("ig2", img2Float);
//	poppy::show_image("mf1", masked1Float);
//	poppy::show_image("mf2", masked2Float);
//	poppy::show_image("c1", combined1);
//	poppy::show_image("c2", combined2);
//	poppy::wait_key();
	combined1.convertTo(reduced1, CV_8UC3, 255);
	combined2.convertTo(reduced2, CV_8UC3, 255);
}

void Extractor::foreground(Mat &foreground1, Mat &foreground2) {
	cerr << "extract foreground..." << endl;
//	Mat reduced1, reduced2;
//	reduceBackground(img1_.clone(), img2_.clone(), reduced1, reduced2);
//	poppy::show_image("r1", reduced1);
//	poppy::show_image("r2", reduced2);

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

	Mat masked1, masked2;
	Mat finalMask1, finalMask2;

	finalMask1Float.convertTo(finalMask1, CV_8U, 255.0);
	finalMask2Float.convertTo(finalMask2, CV_8U, 255.0);
	equalizeHist(finalMask1, finalMask1);
	equalizeHist(finalMask2, finalMask2);

	finalMask1.convertTo(finalMask1Float, CV_32F, 1.0/255.0);
	finalMask2.convertTo(finalMask2Float, CV_32F, 1.0/255.0);
	int logBase = 20;
	Mat logMask(finalMask1Float.size(),CV_32F);
	logMask = Scalar::all(logBase);
	log(logMask, logMask);
	log(finalMask1Float * (logBase - 1.0) + 1.0, finalMask1Float);
	log(finalMask2Float * (logBase - 1.0) + 1.0, finalMask2Float);
	divide(finalMask1Float, logMask, finalMask1Float);
	divide(finalMask2Float, logMask, finalMask2Float);

	dilate( finalMask1Float, finalMask1Float, getStructuringElement( MORPH_ELLIPSE, Size( 11, 11 ),  Point( 5, 5 ) ) );
	dilate( finalMask2Float, finalMask2Float, getStructuringElement( MORPH_ELLIPSE, Size( 11, 11 ),  Point( 5, 5 ) ) );
	GaussianBlur(finalMask1Float, finalMask1Float, {11,11}, 5);
	GaussianBlur(finalMask2Float, finalMask2Float, {11,11}, 5);
	show_image("mk1", finalMask1Float);
	show_image("mk2", finalMask2Float);
	multiply(grey1Float, finalMask1Float, masked1);
	multiply(grey2Float, finalMask2Float, masked2);

	//convert back to 8-bit grey scale
	masked1.convertTo(masked1, CV_8U, 255.0);
	masked2.convertTo(masked2, CV_8U, 255.0);

	//adjust contrast and brightness
//	auto_adjust_contrast_and_brightness(masked1, foreground1, 2);
//	auto_adjust_contrast_and_brightness(masked2, foreground2, 2);
	equalizeHist(masked1, foreground1);
	equalizeHist(masked2, foreground2);
	masked1.release();
	masked2.release();

	show_image("fg1", foreground1);
	show_image("fg2", foreground2);
}

} /* namespace poppy */
