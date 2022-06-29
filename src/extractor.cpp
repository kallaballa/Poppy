#include "extractor.hpp"

#include "settings.hpp"
#include "util.hpp"
#include "draw.hpp"

#include <iostream>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

using std::cerr;
using std::cout;
using std::endl;

namespace poppy {

Extractor::Extractor() {
}

Extractor::~Extractor() {
}

pair<vector<Point2f>, vector<Point2f>> Extractor::extractKeypoints(const Mat &grey1, const Mat &grey2) {
	if (Settings::instance().max_keypoints == -1)
		Settings::instance().max_keypoints = sqrt(grey1.cols * grey1.rows);
	Ptr<ORB> detector = ORB::create(Settings::instance().max_keypoints);

	vector<KeyPoint> keypoints1, keypoints2;

	Mat descriptors1, descriptors2;
	detector->detect(grey1, keypoints1);
	detector->detect(grey2, keypoints2);

	detector->compute(grey1, keypoints1, descriptors1);
	detector->compute(grey2, keypoints2, descriptors2);

//	cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::LshIndexParams(6, 12, 1);
//	FlannBasedMatcher matcher(indexParams);
//	std::vector<std::vector<cv::DMatch>> matches;
//	matcher.knnMatch(descriptors1, descriptors2, matches, 2);
//
//	std::vector<Point2f> points1;
//	std::vector<Point2f> points2;
//
//	for(auto& v : matches) {
//		for(auto& dm : v) {
//			points1.push_back(keypoints1[dm.queryIdx].pt);
//			points2.push_back(keypoints2[dm.trainIdx].pt);
//		}
//	}
//
//	return {points1,points2};

	vector<Point2f> points1, points2;
	for (auto pt1 : keypoints1)
		points1.push_back(pt1.pt);

	for (auto pt2 : keypoints2)
		points2.push_back(pt2.pt);

	if (points1.size() > points2.size())
		points1.resize(points2.size());
	else
		points2.resize(points1.size());

	return {points1,points2};
}

void Extractor::extractForegroundMask(const Mat &grey, Mat &fgMask) {
	// create a foreground mask by blurring the image over again and tracking the flow of pixels.
	fgMask = Mat::ones(grey.rows, grey.cols, grey.type());
	Mat last = grey.clone();
	Mat fgMaskBlur;
	Mat med, flow;

	//optical flow tracking works as well but is much slower
	auto pBackSub1 = createBackgroundSubtractorMOG2();
	for (size_t i = 0; i < 12; ++i) {
		medianBlur(last, med, i * 8 + 1);
		pBackSub1->apply(med, flow);
		fgMask = fgMask + (flow * (1.0 / 6.0));
		GaussianBlur(fgMask, fgMaskBlur, { 23, 23 }, 1);
		fgMask = fgMaskBlur.clone();
		last = med.clone();
		med.release();
		flow.release();
		fgMaskBlur.release();
	}
	last.release();
}

void Extractor::extractContours(const Mat &img1, const Mat &img2, Mat &contourMap1, Mat &contourMap2, vector<Mat>& contourLayers1, vector<Mat>& contourLayers2, Mat& plainContours1, Mat& plainContours2) {
	Mat grey1, grey2;
	vector<vector<vector<Point2f>>> collected1;
	vector<vector<vector<Point2f>>> collected2;

	cvtColor(img1, grey1, COLOR_RGB2GRAY);
	cvtColor(img2, grey2, COLOR_RGB2GRAY);
	equalizeHist(grey1, grey1);
	equalizeHist(grey2, grey2);
	plainContours1 = Mat::zeros(img1.rows, img1.cols, CV_8UC1);
	plainContours2 = Mat::zeros(img1.rows, img1.cols, CV_8UC1);
	Mat edges1;
	Mat edges2;

	Canny( grey1, edges1, 127, 255 );
	vector<Vec4i> h1;
	vector<vector<Point>> c1;
	findContours(edges1, c1, h1, RETR_TREE, CHAIN_APPROX_TC89_KCOS);
	for(size_t i = 0; i < c1.size(); ++i)
			drawContours(plainContours1, c1, i, { 255 }, 1.0, LINE_4, h1, 0);

	Canny( grey2, edges2, 127, 255 );
	vector<Vec4i> h2;
	vector<vector<Point>> c2;
	findContours(edges2, c2, h2, RETR_TREE, CHAIN_APPROX_TC89_KCOS);
	for(size_t i = 0; i < c2.size(); ++i)
			drawContours(plainContours2, c2, i, { 255 }, 1.0, LINE_4, h2, 0);

	show_image("pc1", plainContours1);
	show_image("pc2", plainContours2);

	double t1 = 0;
	double t2 = 255;
	Mat thresh1, thresh2;
	vector<Vec4i> hierarchy1;
	size_t numContours = 0;

	t1 = 0;
	t2 = 0;
	cerr << "thresholding 1" << endl;
	for (off_t i = 0; i < 15; ++i) {
		t1 = max(0, min(255, (int) round((i * 16.0 * Settings::instance().contour_sensitivity))));
		t2 = max(0, min(255, (int) round(((i + 1) * 16.0 * Settings::instance().contour_sensitivity))));
		cerr << t1 << "/" << t2 << '\r';

		threshold(grey1, thresh1, t1, t2, 0);

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
	for (off_t j = 0; j < 15; ++j) {
		t1 = min(255, (int) round((j * 16 * Settings::instance().contour_sensitivity)));
		t2 = min(255, (int) round(((j + 1) * 16 * Settings::instance().contour_sensitivity)));
		cerr << t1 << "/" << t2 << '\r';

		threshold(grey2, thresh2, t1, t2, 0);

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

	contourMap1 = cmap1.clone();
	contourMap2 = cmap2.clone();
	show_image("cmap1", cmap1);
	show_image("cmap2", cmap2);
	assert(contourLayers1.size() == contourLayers2.size());
}

void Extractor::extractForeground(const Mat &img1, const Mat &img2, Mat &foreground1, Mat &foreground2) {
	cerr << "extract features" << endl;
	Mat grey1, grey2, canny1, canny2;
	cvtColor(img1, grey1, COLOR_RGB2GRAY);
	cvtColor(img2, grey2, COLOR_RGB2GRAY);

	Mat fgMask1;
	Mat fgMask2;
	//extract areas of interest (aka. foreground)
	extractForegroundMask(grey1, fgMask1);
	extractForegroundMask(grey2, fgMask2);
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

	grey1.release();
	grey2.release();
	fgMask1.release();
	fgMask2.release();

	//multiply the fg mask with the radial mask to emphasize features in the center of the image
	Mat finalMask1Float, finalMask2Float;
	if (Settings::instance().enable_radial_mask) {
		multiply(fgMask1Float, radialMaskFloat, finalMask1Float);
		multiply(fgMask2Float, radialMaskFloat, finalMask2Float);
	} else {
		fgMask1Float.copyTo(finalMask1Float);
		fgMask2Float.copyTo(finalMask2Float);
	}

	radialMaskFloat.release();
	fgMask1Float.release();
	fgMask2Float.release();

	show_image("mask1", finalMask1Float);
	show_image("mask2", finalMask2Float);
	/*
	 * create the final masked image. uses gaussian blur to sharpen the image.
	 * But before the blurred image is subtracted from the image (to sharpen)
	 * it is divided by the blurred mask. That way features in the center will
	 * be emphasized
	 */
	Mat masked1, masked2;
	Mat blurred1Float, blurredMask1Float, maskedSharp1Float;

	GaussianBlur(grey1Float, blurred1Float, Size(23, 23), 3);
	GaussianBlur(finalMask1Float, blurredMask1Float, Size(23, 23), 3);
	maskedSharp1Float = blurred1Float / blurredMask1Float;
	addWeighted(grey1Float, 1.1, maskedSharp1Float, -0.1, 0, masked1);
	blurred1Float.release();
	blurredMask1Float.release();
	maskedSharp1Float.release();
	grey1Float.release();
	fgMask1Float.release();
	finalMask1Float.release();

	Mat blurred2Float, blurredMask2Float, maskedBlur2Float;
	GaussianBlur(grey2Float, blurred2Float, Size(23, 23), 3);
	GaussianBlur(finalMask2Float, blurredMask2Float, Size(23, 23), 3);
	maskedBlur2Float = blurred2Float / blurredMask2Float;
	addWeighted(grey2Float, 1.1, maskedBlur2Float, -0.1, 0, masked2);
	grey2Float.release();
	fgMask2Float.release();
	blurred2Float.release();
	blurredMask2Float.release();
	maskedBlur2Float.release();
	finalMask2Float.release();

	//convert back to 8-bit grey scale
	masked1.convertTo(masked1, CV_8U, 255.0);
	masked2.convertTo(masked2, CV_8U, 255.0);

	//adjust contrast and brightness
	adjust_contrast_and_brightness(masked1, foreground1, 2, 5);
	adjust_contrast_and_brightness(masked2, foreground2, 2, 5);
	masked1.release();
	masked2.release();

	show_image("fg1", foreground1);
	show_image("fg2", foreground2);
}

} /* namespace poppy */
