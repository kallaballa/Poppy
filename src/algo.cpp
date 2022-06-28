#include "algo.hpp"
#include "draw.hpp"
#include "util.hpp"
#include "blend.hpp"
#include "settings.hpp"
#include "procrustes.hpp"
#include "experiments.hpp"
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/video.hpp>

using namespace std;
using namespace cv;

namespace poppy {

constexpr double RADIANS_TO_DEGREES = 57.2958;

template<typename T>
struct reversion_wrapper {
	T &iterable;
};

template<typename T>
auto begin(reversion_wrapper<T> w) {
	return std::rbegin(w.iterable);
}

template<typename T>
auto end(reversion_wrapper<T> w) {
	return std::rend(w.iterable);
}

template<typename T>
reversion_wrapper<T> reverse(T &&iterable) {
	return {iterable};
}

std::vector<std::vector<Point2f>> convertContourTo2f(const std::vector<std::vector<Point>> &contours1) {
	std::vector<std::vector<Point2f>> tmp;
	for (auto &vec : contours1) {
		std::vector<Point2f> row;
		for (auto &pt : vec) {
			row.push_back(Point2f(pt.x, pt.y));
		}
		if (!row.empty())
			tmp.push_back(row);
	}
	return tmp;
}

std::vector<std::vector<Point>> convertContourFrom2f(const std::vector<std::vector<Point2f>> &contours1) {
	std::vector<std::vector<Point>> tmp;
	for (auto &vec : contours1) {
		std::vector<Point> row;
		for (auto &pt : vec) {
			row.push_back(Point(pt.x, pt.y));
		}
		if (!row.empty())
			tmp.push_back(row);
	}
	return tmp;
}

void canny_threshold(const Mat &src, Mat &detected_edges, double thresh) {
	detected_edges = src.clone();
	Canny(detected_edges, detected_edges, thresh, thresh * 2);
}

void make_delaunay_mesh(const Size &size, Subdiv2D &subdiv, vector<Point2f> &dstPoints) {
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point2f> pt(3);
	Rect rect(0, 0, size.width, size.height);

	for (size_t i = 0; i < triangleList.size(); i++) {
		Vec6f t = triangleList[i];
		pt[0] = Point2f(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point2f(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point2f(cvRound(t[4]), cvRound(t[5]));

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
			dstPoints.push_back(pt[0]);
			dstPoints.push_back(pt[1]);
			dstPoints.push_back(pt[2]);
		}
	}
}

pair<vector<Point2f>, vector<Point2f>> find_keypoints(const Mat &grey1, const Mat &grey2) {
	if (Settings::instance().max_keypoints == -1)
		Settings::instance().max_keypoints = sqrt(grey1.cols * grey1.rows);
	Ptr<ORB> detector = ORB::create(Settings::instance().max_keypoints);
//	Ptr<ORB> extractor = ORB::create();

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

Mat points_to_homogenous_mat(const vector<Point> &pts) {
	int numPts = pts.size();
	Mat homMat(3, numPts, CV_32FC1);
	for (int i = 0; i < numPts; i++) {
		homMat.at<float>(0, i) = pts[i].x;
		homMat.at<float>(1, i) = pts[i].y;
		homMat.at<float>(2, i) = 1.0;
	}
	return homMat;
}

void morph_points(vector<Point2f> &srcPts1, vector<Point2f> &srcPts2, vector<Point2f> &dstPts, float s) {
	assert(srcPts1.size() == srcPts2.size());
	int numPts = srcPts1.size();
	double totalDistance = 0;
	dstPts.resize(numPts);
	for (int i = 0; i < numPts; i++) {
		totalDistance += hypot(srcPts2[i].x - srcPts1[i].x, srcPts2[i].y - srcPts1[i].y);
		dstPts[i].x = round((1.0 - s) * srcPts1[i].x + s * srcPts2[i].x);
		dstPts[i].y = round((1.0 - s) * srcPts1[i].y + s * srcPts2[i].y);
	}
}

void get_triangle_indices(const Subdiv2D &subDiv, const vector<Point2f> &points, vector<Vec3i> &triangleVertices) {
	vector<Vec6f> triangles;
	subDiv.getTriangleList(triangles);

	int numTriangles = triangles.size();
	triangleVertices.clear();
	triangleVertices.reserve(numTriangles);
	for (int i = 0; i < numTriangles; i++) {
		vector<Point2f>::const_iterator vert1, vert2, vert3;
		vert1 = find(points.begin(), points.end(), Point2f(triangles[i][0], triangles[i][1]));
		vert2 = find(points.begin(), points.end(), Point2f(triangles[i][2], triangles[i][3]));
		vert3 = find(points.begin(), points.end(), Point2f(triangles[i][4], triangles[i][5]));

		Vec3i vertex;
		if (vert1 != points.end() && vert2 != points.end() && vert3 != points.end()) {
			vertex[0] = vert1 - points.begin();
			vertex[1] = vert2 - points.begin();
			vertex[2] = vert3 - points.begin();
			triangleVertices.push_back(vertex);
		}
	}
}

void make_triangler_points(const vector<Vec3i> &triangleVertices, const vector<Point2f> &points, vector<vector<Point>> &trianglerPts) {
	int numTriangles = triangleVertices.size();
	trianglerPts.resize(numTriangles);
	for (int i = 0; i < numTriangles; i++) {
		vector<Point> triangle;
		for (int j = 0; j < 3; j++) {
			triangle.push_back(Point(points[triangleVertices[i][j]].x, points[triangleVertices[i][j]].y));
		}
		trianglerPts[i] = triangle;
	}
}

void paint_triangles(Mat &img, const vector<vector<Point>> &triangles) {
	int numTriangles = triangles.size();

	for (int i = 0; i < numTriangles; i++) {
		vector<Point> poly(3);

		for (int j = 0; j < 3; j++) {
			poly[j] = Point(cvRound(triangles[i][j].x), cvRound(triangles[i][j].y));
		}
		fillConvexPoly(img, poly, Scalar(i + 1));
	}
}

void solve_homography(const vector<Point> &srcPts1, const vector<Point> &srcPts2, Mat &homography) {
	assert(srcPts1.size() == srcPts2.size());
	homography = points_to_homogenous_mat(srcPts2) * points_to_homogenous_mat(srcPts1).inv();
}

void solve_homography(const vector<vector<Point>> &srcPts1,
		const vector<vector<Point>> &srcPts2,
		vector<Mat> &hmats) {
	assert(srcPts1.size() == srcPts2.size());

	int ptsNum = srcPts1.size();
	hmats.clear();
	hmats.reserve(ptsNum);
	for (int i = 0; i < ptsNum; i++) {
		Mat homography;
		solve_homography(srcPts1[i], srcPts2[i], homography);
		hmats.push_back(homography);
	}
}

void morph_homography(const Mat &Hom, Mat &MorphHom1, Mat &MorphHom2, float blend_ratio) {
	Mat invHom = Hom.inv();
	MorphHom1 = Mat::eye(3, 3, CV_32FC1) * (1.0 - blend_ratio) + Hom * blend_ratio;
	MorphHom2 = Mat::eye(3, 3, CV_32FC1) * blend_ratio + invHom * (1.0 - blend_ratio);
}

void morph_homography(const vector<Mat> &homs,
		vector<Mat> &morphHoms1,
		vector<Mat> &morphHoms2,
		float blend_ratio) {
	int numHoms = homs.size();
	morphHoms1.resize(numHoms);
	morphHoms2.resize(numHoms);
	for (int i = 0; i < numHoms; i++) {
		morph_homography(homs[i], morphHoms1[i], morphHoms2[i], blend_ratio);
	}
}

void create_map(const Mat &triangleMap, const vector<Mat> &homMatrices, Mat &mapx, Mat &mapy) {
	assert(triangleMap.type() == CV_32SC1);

	// Allocate Mat for the map
	mapx.create(triangleMap.size(), CV_32FC1);
	mapy.create(triangleMap.size(), CV_32FC1);

	// Compute inverse matrices
	vector<Mat> invHomMatrices(homMatrices.size());
	for (size_t i = 0; i < homMatrices.size(); i++) {
		invHomMatrices[i] = homMatrices[i].inv();
	}

	for (int y = 0; y < triangleMap.rows; y++) {
		for (int x = 0; x < triangleMap.cols; x++) {
			int idx = triangleMap.at<int>(y, x) - 1;
			if (idx >= 0) {
				Mat H = invHomMatrices[triangleMap.at<int>(y, x) - 1];
				float z = H.at<float>(2, 0) * x + H.at<float>(2, 1) * y + H.at<float>(2, 2);
				if (z == 0)
					z = 0.00001;
				mapx.at<float>(y, x) = (H.at<float>(0, 0) * x + H.at<float>(0, 1) * y + H.at<float>(0, 2)) / z;
				mapy.at<float>(y, x) = (H.at<float>(1, 0) * x + H.at<float>(1, 1) * y + H.at<float>(1, 2)) / z;
			}
			else {
				mapx.at<float>(y, x) = x;
				mapy.at<float>(y, x) = y;
			}
		}
	}
}

void draw_contour_map(Mat &dst, vector<Mat>& contourLayers, const vector<vector<vector<Point2f>>> &collected, const vector<Vec4i> &hierarchy, int cols, int rows, int type) {
	dst = Mat::zeros(rows, cols, type);
	size_t cnt = 0;
	for (size_t i = 0; i < collected.size(); ++i) {
		auto &contours = collected[i];
		double shade = 32.0 + 223.0 * (double(i) / collected.size());

		cnt += contours.size();
		cerr << i << "/" << (collected.size() - 1) << '\r';
		vector<vector<Point>> tmp = convertContourFrom2f(contours);
		Mat layer = Mat::zeros(rows, cols, type);
		for (size_t j = 0; j < tmp.size(); ++j) {
			drawContours(layer, tmp, j, { shade }, 1.0, LINE_4, hierarchy, 0);
			drawContours(dst, tmp, j, { shade }, 1.0, LINE_4, hierarchy, 0);
		}
		contourLayers.push_back(layer);
	}
	cerr << endl;
}

void adjust_contrast_and_brightness(const Mat &src, Mat &dst, double contrast, double lowcut) {
	dst = src.clone();
	int minV = numeric_limits<int>::max();
	int maxV = numeric_limits<int>::min();
	double val = 0;
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			val = double(dst.at<uchar>(y, x));
			maxV = max(double(maxV), val);
			minV = min(double(minV), val);
		}
	}

	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			val = dst.at<uchar>(y, x);
			dst.at<uchar>(y, x) = ((val - minV) / (maxV - minV)) * 255;
		}
	}

	Mat hc(dst);
	Scalar imgAvgVec = sum(dst) / (dst.cols * dst.rows);
	double imgAvg = imgAvgVec[0];
	int brightness = -((contrast - 1) * imgAvg);
	dst.convertTo(hc, -1, contrast, brightness);
	hc.copyTo(dst);
}

double feature_metric(const Mat &grey1) {
	Mat corners;
	cornerHarris(grey1, corners, 2, 3, 0.04);
	cv::Scalar mean, stddev;
	cv::meanStdDev(corners, mean, stddev);

	return stddev[0];
}

void decimate_features(const Mat &img1, const Mat &img2, Mat &decimated1, Mat &decimated2) {
	Mat grey1, grey2;

	cvtColor(img1, grey1, COLOR_RGB2GRAY);
	cvtColor(img2, grey2, COLOR_RGB2GRAY);
	cerr << "decimate features" << endl;

	decimated1 = img1.clone();
	decimated2 = img2.clone();

	double cnz1 = feature_metric(grey1);
	double cnz2 = feature_metric(grey2);
	cerr << "features 1: " << cnz1 << endl;
	cerr << "features 2: " << cnz2 << endl;

//	while ((cnz1 / maxSum) > 0.5) {
//		Mat blurred1, lap1, blurredMask;
//		medianBlur(decimated1, blurred1, 23);
//		Laplacian(blurred1, lap1, blurred1.depth());
//		decimated1 = blurred1 - (0.7 * lap1);
//		cnz1 = sum(decimated1)[0];
//		cerr << "decimate features 1: " << cnz1 << endl;
//	}

//	while ((cnz2 / maxSum) > 0.5) {
//		Mat blurred2, lap2, blurredMask;
//		medianBlur(decimated2, blurred2, 23);
//		Laplacian(blurred2, lap2, blurred2.depth());
//		decimated2 = blurred2 - (0.7 * lap2);
//		cnz2 = sum(decimated2)[0];
//		cerr << "decimate features 2: " << cnz2 << endl;
//	}

	adjust_contrast_and_brightness(decimated1, decimated1, 2, 5);
	adjust_contrast_and_brightness(decimated2, decimated2, 2, 5);
	int i = 0;
	while (cnz1 > 0.001) {
		Mat blurred1;
		GaussianBlur(decimated1, blurred1, {i * 2 + 1,i * 2 + 1}, i);
		decimated1 = blurred1.clone();
		cvtColor(decimated1, grey1, COLOR_RGB2GRAY);
		cnz1 = feature_metric(grey1);
		cerr << "redecimate features 1: " << cnz1 << "\r";
		++i;
	}

	i = 0;
	while (cnz2 > 0.001) {
		Mat blurred2;
		GaussianBlur(decimated2, blurred2, {i * 2 + 1,i * 2 + 1}, i);
		decimated2 = blurred2.clone();
		cvtColor(decimated2, grey2, COLOR_RGB2GRAY);
		cnz2 = feature_metric(grey2);
		cerr << "redecimate features 2: " << cnz2 << "\r";
		++i;
	}

	if (cnz2 < (cnz1 * 0.9)) {
		int i = 0;
		while (cnz2 < (cnz1 * 0.9)) {
			Mat blurred1;
			GaussianBlur(decimated1, blurred1, { i * 2 + 1, i * 2 + 1 }, i);
			decimated1 = blurred1.clone();
			cvtColor(decimated1, grey1, COLOR_RGB2GRAY);
			cnz1 = feature_metric(grey1);
			cerr << "redecimate features 1: " << cnz1 << "\r";
			++i;
		}
		cerr << endl;
	} else if (cnz1 < (cnz2 * 0.9)) {
		int i = 0;
		while (cnz1 < (cnz2 * 0.9)) {
			Mat blurred2;
			GaussianBlur(decimated2, blurred2, { i * 2 + 1, i * 2 + 1 }, i);
			decimated2 = blurred2.clone();
			cvtColor(decimated2, grey2, COLOR_RGB2GRAY);
			cnz2 = feature_metric(grey2);
			cerr << "redecimate features 2: " << cnz2 << "\r";
			++i;
		}
		cerr << endl;
	}

//	show_image("dec1", decimated1);
//	show_image("dec2", decimated2);
}

void extract_foreground_mask(const Mat &grey, Mat &fgMask) {
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

void extract_contours(const Mat &img1, const Mat &img2, Mat &contourMap1, Mat &contourMap2, vector<Mat>& contourLayers1, vector<Mat>& contourLayers2, Mat& plainContours1, Mat& plainContours2) {
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

void extract_foreground(const Mat &img1, const Mat &img2, Mat &foreground1, Mat &foreground2) {
	cerr << "extract features" << endl;
	Mat grey1, grey2, canny1, canny2;
	cvtColor(img1, grey1, COLOR_RGB2GRAY);
	cvtColor(img2, grey2, COLOR_RGB2GRAY);

	Mat fgMask1;
	Mat fgMask2;
	//extract areas of interest (aka. foreground)
	extract_foreground_mask(grey1, fgMask1);
	extract_foreground_mask(grey2, fgMask2);
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

pair<double, Point2f> get_orientation(const vector<Point2f> &pts)
		{
	//Construct a buffer used by the pca analysis
	int sz = static_cast<int>(pts.size());
	Mat data_pts = Mat(sz, 2, CV_64F);
	for (int i = 0; i < data_pts.rows; i++)
			{
		data_pts.at<double>(i, 0) = pts[i].x;
		data_pts.at<double>(i, 1) = pts[i].y;
	}
	//Perform PCA analysis
	PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);
	//Store the center of the object
	Point2f cntr = Point2f(pca_analysis.mean.at<double>(0, 0), pca_analysis.mean.at<double>(0, 1));
	//Store the eigenvalues and eigenvectors
	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; i++) {
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0), pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
	}

	return {atan2(eigen_vecs[0].y, eigen_vecs[0].x), cntr};
}

void translate(const Mat &src, Mat &dst, const Point2f& by) {
	float warpValues[] = { 1.0, 0.0, by.x, 0.0, 1.0, by.y };
	Mat translation_matrix = Mat(2, 3, CV_32F, warpValues);
	warpAffine(src, dst, translation_matrix, src.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
}

void rotate(const Mat &src, Mat &dst, Point2f center, double angle, double scale) {
	Mat rm = getRotationMatrix2D(center, angle, scale);
	warpAffine(src, dst, rm, src.size());
}

Point2f rotate_point(const cv::Point2f &inPoint, const double &angDeg) {
	double rad = angDeg * M_PI / 180.0;
	cv::Point2f outPoint;
	outPoint.x = std::cos(rad) * inPoint.x - std::sin(rad) * inPoint.y;
	outPoint.y = std::sin(rad) * inPoint.x + std::cos(rad) * inPoint.y;
	return outPoint;
}

Point2f rotate_point(const cv::Point2f &inPoint, const cv::Point2f &center, const double &angDeg) {
	return rotate_point(inPoint - center, angDeg) + center;
}

void translate_points(vector<Point2f> &pts, const Point2f &by) {
	for (auto &pt : pts) {
		pt += by;
	}
}

void rotate_points(vector<Point2f> &pts, const Point2f &center, const double &angDeg) {
	for (auto &pt : pts) {
		pt = rotate_point(pt - center, angDeg) + center;
	}
}

void scale_points(vector<Point2f> &pts, double coef) {
	for (auto &pt : pts) {
		pt.x *= coef;
		pt.y *= coef;
	}
}

void scale_features(Features &ft, double coef) {
	scale_points(ft.chin_, coef);
	scale_points(ft.top_nose_, coef);
	scale_points(ft.bottom_nose_, coef);
	scale_points(ft.left_eyebrow_, coef);
	scale_points(ft.right_eyebrow_, coef);
	scale_points(ft.left_eye_, coef);
	scale_points(ft.right_eye_, coef);
	scale_points(ft.outer_lips_, coef);
	scale_points(ft.inside_lips_, coef);
}

void plot(Mat &img, vector<Point2f> points, Scalar color, int radius = 2) {
	for (Point2f p : points)
		circle(img, p, radius, color, radius * 2);
}

void retranslate(Mat &corrected2, Mat &contourMap2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, const size_t width, const size_t height) {
	vector<Point2f> left;
	vector<Point2f> right;
	vector<Point2f> top;
	vector<Point2f> bottom;
	for (auto &pt : srcPoints2) {
		left.push_back( { pt.x - 1, pt.y });
		right.push_back( { pt.x + 1, pt.y });
		top.push_back( { pt.x, pt.y - 1 });
		bottom.push_back( { pt.x, pt.y + 1 });
	}
	double mdCurrent = morph_distance(srcPoints1, srcPoints2, width, height);
	double mdLeft = morph_distance(srcPoints1, left, width, height);
	double mdRight = morph_distance(srcPoints1, right, width, height);
	double mdTop = morph_distance(srcPoints1, top, width, height);
	double mdBottom = morph_distance(srcPoints1, bottom, width, height);
	off_t xchange = 0;
	off_t ychange = 0;

	if (mdLeft < mdCurrent)
		xchange = -1;
	else if (mdRight < mdCurrent)
		xchange = +1;

	if (mdTop < mdCurrent)
		ychange = -1;
	else if (mdBottom < mdCurrent)
		ychange = +1;
	cerr << "current morph dist: " << mdCurrent << endl;
	off_t xProgress = 1;

	if (xchange != 0) {
		double lastMorphDist = mdCurrent;
		double morphDist = 0;
		do {
			vector<Point2f> tmp;
			for (auto &pt : srcPoints2) {
				tmp.push_back( { pt.x + xchange * xProgress, pt.y });
			}
			morphDist = morph_distance(srcPoints1, tmp, width, height);
//			cerr << "morph dist x: " << morphDist << endl;
			if (morphDist > lastMorphDist)
				break;
			lastMorphDist = morphDist;
			++xProgress;
		} while (true);
	}
	off_t yProgress = 1;

	if (ychange != 0) {
		double lastMorphDist = mdCurrent;
		double morphDist = 0;

		do {
			vector<Point2f> tmp;
			for (auto &pt : srcPoints2) {
				tmp.push_back( { pt.x, pt.y + ychange * yProgress});
			}
			morphDist = morph_distance(srcPoints1, tmp, width, height);
//			cerr << "morph dist y: " << morphDist << endl;
			if (morphDist > lastMorphDist)
				break;
			lastMorphDist = morphDist;
			++yProgress;
		} while (true);
	}
	Point2f retranslation(xchange * xProgress, ychange * yProgress);
	cerr << "retranslation: " << retranslation << endl;
	translate(corrected2, corrected2, retranslation);
	translate(contourMap2, contourMap2, retranslation);
	for (auto &pt : srcPoints2) {
		pt.x += retranslation.x;
		pt.y += retranslation.y;
	}
}

void rerotate(Mat &corrected2, Mat &contourMap2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, const size_t width, const size_t height) {
	double morphDist = -1;
	vector<Point2f> tmp;
	Point2f center = {float(corrected2.cols/2.0), float(corrected2.cols/2.0)};
	double lowestDist = numeric_limits<double>::max();
	double selectedAngle = 0;
	for(size_t i = 0; i < 3600; ++i) {
		tmp = srcPoints2;
		rotate_points(tmp, center, i / 10.0);
		morphDist = morph_distance(srcPoints1, tmp, width, height);
		if(morphDist < lowestDist) {
			lowestDist = morphDist;
			selectedAngle = i / 10.0;
		}
	}

	cerr << "dist: " << lowestDist << " angle: " << selectedAngle << "°" << endl;
	rotate(corrected2, corrected2, center, -selectedAngle);
	rotate(contourMap2, contourMap2, center, -selectedAngle);
	rotate_points(srcPoints2, center, -selectedAngle);
}

void find_matches(const Mat &orig1, const Mat &orig2, Features& ft1, Features& ft2, Mat &corrected1, Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, Mat &contourMap1, Mat &contourMap2) {
	if (ft1.empty() || ft2.empty()) {
		cerr << "general algorithm..." << endl;
		Mat goodFeatures1, goodFeatures2;
		extract_foreground(orig1, orig2, goodFeatures1, goodFeatures2);
		vector<Mat> contourLayers1;
		vector<Mat> contourLayers2;
		Mat plainContours1;
		Mat plainContours2;
		extract_contours(orig1, orig2, contourMap1, contourMap2, contourLayers1, contourLayers2, plainContours1, plainContours2);
		corrected1 = orig1.clone();
		corrected2 = orig2.clone();

		show_image("gf1", goodFeatures1);
		show_image("gf2", goodFeatures2);

		cerr << "find matches..." << endl;

		if (Settings::instance().enable_auto_align) {
			cerr << "auto aligning..." << endl;

			srcPoints1.clear();
			srcPoints2.clear();
			auto matches = find_keypoints(goodFeatures1, goodFeatures2);
			srcPoints1.insert(srcPoints1.end(), matches.first.begin(), matches.first.end());
			srcPoints2.insert(srcPoints2.end(), matches.second.begin(), matches.second.end());
			cerr << "collected keypoints: " << srcPoints1.size() << "/" << srcPoints2.size() << endl;

			retranslate(corrected2, contourMap2, srcPoints1, srcPoints2, contourMap1.cols, contourMap1.rows);
			rerotate(corrected2, contourMap2, srcPoints1, srcPoints2, contourMap1.cols, contourMap1.rows);
		} else {
			srcPoints1.clear();
			srcPoints2.clear();

			auto matches = find_keypoints(goodFeatures1, goodFeatures2);

			srcPoints1.insert(srcPoints1.end(), matches.first.begin(), matches.first.end());
			srcPoints2.insert(srcPoints2.end(), matches.second.begin(), matches.second.end());
		}
	} else {
		cerr << "face algorithm..." << endl;
		assert(!ft1.empty() && !ft2.empty());
		vector<Mat> contourLayers1;
		vector<Mat> contourLayers2;
		Mat plainContours1;
		Mat plainContours2;
		extract_contours(orig1, orig2, contourMap1, contourMap2, contourLayers1, contourLayers2, plainContours1, plainContours2);

		if (Settings::instance().enable_auto_align) {
			cerr << "auto aligning..." << endl;

			double w1 = fabs(ft1.right_eye_[0].x - ft1.left_eye_[0].x);
			double w2 = fabs(ft2.right_eye_[0].x - ft2.left_eye_[0].x);
			double scale2 = w1 / w2;
			Mat scaled2;
			resize(orig2, scaled2, Size { int(std::round(orig2.cols * scale2)), int(std::round(orig2.rows * scale2)) });
			srcPoints1 = ft1.getAllPoints();

			Point2f eyeVec1 = ft1.right_eye_[0] - ft1.left_eye_[0];
			Point2f eyeVec2 = ft2.right_eye_[0] - ft2.left_eye_[0];
			Point2f center1(ft1.left_eye_[0].x + (eyeVec1.x / 2.0), ft1.left_eye_[0].y + (eyeVec1.y / 2.0));
			Point2f center2(ft2.left_eye_[0].x + (eyeVec2.x / 2.0), ft2.left_eye_[0].y + (eyeVec2.y / 2.0));
			double angle1 = atan2(eyeVec1.y, eyeVec1.x);
			double angle2 = atan2(eyeVec2.y, eyeVec2.x);
			double dy = center1.y - center2.y;
			double dx = center1.x - center2.x;

			Mat translated2;
			translate(scaled2, translated2, {float(dx), float(dy)});

			angle1 = angle1 * 180 / M_PI;
			angle2 = angle2 * 180 / M_PI;
			angle1 = angle1 < 0 ? angle1 + 360 : angle1;
			angle2 = angle2 < 0 ? angle2 + 360 : angle2;
			std::cerr << "angle: " << angle1 << "/" << angle2 << endl;
			double targetAng = angle2 - angle1;
			Mat rotated2;
			rotate(translated2, rotated2, center2, targetAng);
			ft2 = FaceDetector::instance().detect(rotated2);
			srcPoints2 = ft2.getAllPoints();

			double dw = rotated2.cols - orig2.cols;
			double dh = rotated2.rows - orig2.rows;
			corrected2 = rotated2(Rect(dw / 2, dh / 2, orig2.cols, orig2.rows));
			corrected1 = orig1.clone();
			assert(corrected1.cols == corrected2.cols && corrected1.rows == corrected2.rows);
		} else {
			srcPoints1 = ft1.getAllPoints();
			srcPoints2 = ft2.getAllPoints();

			corrected1 = orig1.clone();
			corrected2 = orig2.clone();
		}
	}
	filter_invalid_points(srcPoints1, srcPoints2, orig1.cols, orig1.rows);

	cerr << "keypoints: " << srcPoints1.size() << "/" << srcPoints2.size() << endl;
	check_points(srcPoints1, orig1.cols, orig1.rows);
	check_points(srcPoints2, orig1.cols, orig1.rows);
}

tuple<double, double, double> calculate_sum_mean_and_sd(multimap<double, pair<Point2f, Point2f>> distanceMap) {
	size_t s = distanceMap.size();
	double sum = 0.0, mean, standardDeviation = 0.0;

	for (auto &p : distanceMap) {
		sum += p.first;
	}

	mean = sum / s;

	for (auto &p : distanceMap) {
		standardDeviation += pow(p.first - mean, 2);
	}

	return {sum, mean, sqrt(standardDeviation / s)};
}
std::random_device rd;
std::mt19937 g(rd());

void match_points_by_proximity(vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, int cols, int rows) {
	multimap<double, pair<Point2f, Point2f>> distanceMap;
	std::shuffle(srcPoints1.begin(), srcPoints1.end(), g);
	std::shuffle(srcPoints2.begin(), srcPoints2.end(), g);

	Point2f nopoint(-1, -1);
	for (auto &pt1 : srcPoints1) {
		double dist = 0;
		double currentMinDist = numeric_limits<double>::max();

		Point2f *closest = &nopoint;
		for (auto &pt2 : srcPoints2) {
			if (pt2.x == -1 && pt2.y == -1)
				continue;

			dist = hypot(pt2.x - pt1.x, pt2.y - pt1.y);

			if (dist < currentMinDist) {
				currentMinDist = dist;
				closest = &pt2;
			}
		}
		if (closest->x == -1 && closest->y == -1)
			continue;

		dist = hypot(closest->x - pt1.x, closest->y - pt1.y);
		distanceMap.insert( { dist, { pt1, *closest } });
		closest->x = -1;
		closest->y = -1;
	}
	assert(srcPoints1.size() == srcPoints2.size());
	assert(!srcPoints1.empty() && !srcPoints2.empty());

	auto distribution = calculate_sum_mean_and_sd(distanceMap);
	assert(!distanceMap.empty());
	srcPoints1.clear();
	srcPoints2.clear();

	double mean = get<1>(distribution);
	double deviation = get<2>(distribution);
	if(mean == 0) {
		for (auto it = distanceMap.rbegin(); it != distanceMap.rend(); ++it) {
			srcPoints1.push_back((*it).second.first);
			srcPoints2.push_back((*it).second.second);
		}
		assert(srcPoints1.size() == srcPoints2.size());
		assert(!srcPoints1.empty() && !srcPoints2.empty());

		return;
	}
	double value = 0;
	double limit = mean;
	double limitCoef = 0.9;
	do {
		srcPoints1.clear();
		srcPoints2.clear();
		for (auto it = distanceMap.rbegin(); it != distanceMap.rend(); ++it) {
			value = (*it).first;

			if ((value > 0 && value < limit)) {
				srcPoints1.push_back((*it).second.first);
				srcPoints2.push_back((*it).second.second);
			}
		}

		assert(srcPoints1.size() == srcPoints2.size());
		check_points(srcPoints1, cols, rows);
		check_points(srcPoints2, cols, rows);
		if(srcPoints1.empty()) {
			limitCoef *= 1.1;
			limit *= (1.0 / limitCoef);
			continue;
		} else
			limit *= limitCoef;
	} while (srcPoints1.empty() || srcPoints1.size() > (distanceMap.size() / (16.0 / Settings::instance().match_tolerance)));
	bool compares = false;
	for(size_t i = 0; i < srcPoints1.size(); ++i) {
		if(!(compares = (srcPoints1[i] == srcPoints2[i])))
			break;
	}

	assert(srcPoints1.size() == srcPoints2.size());
	assert(!srcPoints1.empty() && !srcPoints2.empty());
}


void add_corners(vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, MatSize sz) {
	float w = sz().width - 1;
	float h = sz().height - 1;
	srcPoints1.push_back(Point2f(0, 0));
	srcPoints1.push_back(Point2f(w, 0));
	srcPoints1.push_back(Point2f(0, h));
	srcPoints1.push_back(Point2f(w, h));
	srcPoints2.push_back(Point2f(0, 0));
	srcPoints2.push_back(Point2f(w, 0));
	srcPoints2.push_back(Point2f(0, h));
	srcPoints2.push_back(Point2f(w, h));
}

void prepare_matches(Mat &src1, Mat &src2, const Mat &img1, const Mat &img2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2) {
	//edit matches
	cerr << "prepare: " << srcPoints1.size() << endl;
	match_points_by_proximity(srcPoints1, srcPoints2, img1.cols, img1.rows);
	cerr << "match: " << srcPoints1.size() << endl;

	Mat matMatches;
	Mat grey1, grey2;
	cvtColor(src1, grey1, COLOR_RGB2GRAY);
	cvtColor(src2, grey2, COLOR_RGB2GRAY);
	draw_matches(grey1, grey2, matMatches, srcPoints1, srcPoints2);
	show_image("matched", matMatches);

	if (srcPoints1.size() > srcPoints2.size())
		srcPoints1.resize(srcPoints2.size());
	else
		srcPoints2.resize(srcPoints1.size());

	add_corners(srcPoints1, srcPoints2, src1.size);
}

double morph_images(const Mat &origImg1, const Mat &origImg2, Mat &dst, const Mat &last, vector<Point2f> &morphedPoints, vector<Point2f> srcPoints1, vector<Point2f> srcPoints2, Mat &allContours1, Mat &allContours2, double shapeRatio, double maskRatio) {
	Size SourceImgSize(origImg1.cols, origImg1.rows);
	Subdiv2D subDiv1(Rect(0, 0, SourceImgSize.width, SourceImgSize.height));
	Subdiv2D subDiv2(Rect(0, 0, SourceImgSize.width, SourceImgSize.height));
	Subdiv2D subDivMorph(Rect(0, 0, SourceImgSize.width, SourceImgSize.height));

	vector<Point2f> uniq1, uniq2, uniqMorph;
	clip_points(srcPoints1, origImg1.cols, origImg1.rows);
	check_points(srcPoints1, origImg1.cols, origImg1.rows);
	make_uniq(srcPoints1, uniq1);
	check_uniq(uniq1);
	subDiv1.insert(uniq1);
//	for (auto pt : uniq1)
//		subDiv1.insert(pt);

	clip_points(srcPoints2, origImg2.cols, origImg2.rows);
	check_points(srcPoints2, origImg2.cols, origImg2.rows);
	make_uniq(srcPoints2, uniq2);
	check_uniq(uniq2);
	subDiv2.insert(uniq2);
//	for (auto pt : uniq2)
//		subDiv2.insert(pt);

	morph_points(srcPoints1, srcPoints2, morphedPoints, shapeRatio);
	assert(srcPoints1.size() == srcPoints2.size() && srcPoints2.size() == morphedPoints.size());

	clip_points(morphedPoints, origImg1.cols, origImg1.rows);
	check_points(morphedPoints, origImg1.cols, origImg1.rows);
	make_uniq(morphedPoints, uniqMorph);
	check_uniq(uniqMorph);
	subDivMorph.insert(uniqMorph);
	//	for (auto pt : uniqMorph)
//		subDivMorph.insert(pt);

// Get the ID list of corners of Delaunay traiangles.
	vector<Vec3i> triangleIndices;
	get_triangle_indices(subDivMorph, morphedPoints, triangleIndices);

	// Get coordinates of Delaunay corners from ID list
	vector<vector<Point>> triangleSrc1, triangleSrc2, triangleMorph;
	make_triangler_points(triangleIndices, srcPoints1, triangleSrc1);
	make_triangler_points(triangleIndices, srcPoints2, triangleSrc2);
	make_triangler_points(triangleIndices, morphedPoints, triangleMorph);

	// Create a map of triangle ID in the morphed image.
	Mat triMap = Mat::zeros(SourceImgSize, CV_32SC1);
	paint_triangles(triMap, triangleMorph);

	// Compute Homography matrix of each triangle.
	vector<Mat> homographyMats, morphHom1, morphHom2;
	solve_homography(triangleSrc1, triangleSrc2, homographyMats);
	morph_homography(homographyMats, morphHom1, morphHom2, shapeRatio);

	Mat trImg1;
	Mat trans_map_x1, trans_map_y1;
	create_map(triMap, morphHom1, trans_map_x1, trans_map_y1);
	remap(origImg1, trImg1, trans_map_x1, trans_map_y1, INTER_LINEAR);

	Mat trImg2;
	Mat trMapX2, trMapY2;
	create_map(triMap, morphHom2, trMapX2, trMapY2);
	remap(origImg2, trImg2, trMapX2, trMapY2, INTER_LINEAR);

	homographyMats.clear();
	morphHom1.clear();
	morphHom2.clear();
	triMap.release();

	Mat_<Vec3f> l;
	Mat_<Vec3f> r;
	trImg1.convertTo(l, CV_32F, 1.0 / 255.0);
	trImg2.convertTo(r, CV_32F, 1.0 / 255.0);
	Mat_<float> m(l.rows, l.cols, 0.0);
	Mat_<float> m1(l.rows, l.cols, 0.0);
	Mat_<float> m2(l.rows, l.cols, 0.0);
	equalizeHist(allContours1, allContours1);
	equalizeHist(allContours2, allContours2);

	for (off_t x = 0; x < m1.cols; ++x) {
		for (off_t y = 0; y < m1.rows; ++y) {
			m1.at<float>(y, x) = allContours1.at<uint8_t>(y, x) / 255.0;
		}
	}

	for (off_t x = 0; x < m2.cols; ++x) {
		for (off_t y = 0; y < m2.rows; ++y) {
			m2.at<float>(y, x) = allContours2.at<uint8_t>(y, x) / 255.0;
		}
	}

	Mat ones = Mat::ones(m1.rows, m1.cols, m1.type());
	Mat mask;
	m = ones * (1.0 - maskRatio) + m2 * maskRatio;
	show_image("blend", m);
	off_t kx = round(m.cols / 32.0);
	off_t ky = round(m.rows / 32.0);
	if (kx % 2 != 1)
		kx -= 1;

	if (ky % 2 != 1)
		ky -= 1;
	GaussianBlur(m, mask, Size(kx, ky), 12);

	LaplacianBlending lb(l, r, mask, Settings::instance().pyramid_levels);
	Mat_<Vec3f> lapBlend = lb.blend();
	lapBlend.convertTo(dst, origImg1.depth(), 255.0);
	Mat analysis = dst.clone();
	Mat prev = last.clone();
	if (prev.empty())
		prev = dst.clone();
	draw_morph_analysis(dst, prev, analysis, SourceImgSize, subDiv1, subDiv2, subDivMorph, { 0, 0, 255 });
	show_image("mesh", analysis);
//	wait_key();
	return 0;
}
}
