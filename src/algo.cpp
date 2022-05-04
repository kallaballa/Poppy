#include "algo.hpp"
#include "draw.hpp"
#include "util.hpp"
#include "blend.hpp"
#include "settings.hpp"
#include "face.hpp"
#include <iostream>
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

void canny_threshold(const Mat &src, Mat &detected_edges, double thresh) {
	detected_edges = src.clone();
	Canny(detected_edges, detected_edges, thresh, thresh * 2);
}

void make_delaunay_mesh(const Size &size, Subdiv2D &subdiv, vector<Point2f> &dstPoints) {
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);
	Rect rect(0, 0, size.width, size.height);

	for (size_t i = 0; i < triangleList.size(); i++) {
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
			dstPoints.push_back(pt[0]);
			dstPoints.push_back(pt[1]);
			dstPoints.push_back(pt[2]);
		}
	}
}

pair<vector<Point2f>, vector<Point2f>> find_keypoints(const Mat &grey1, const Mat &grey2) {
	if (Settings::instance().max_keypoints == -1)
		Settings::instance().max_keypoints = hypot(grey1.cols, grey1.rows) * 5.0;
	Ptr<ORB> detector = ORB::create(Settings::instance().max_keypoints);
	Ptr<ORB> extractor = ORB::create();

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

Mat points_to_homogenous_mat(const vector<Point2f> &pts) {
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

void make_triangler_points(const vector<Vec3i> &triangleVertices, const vector<Point2f> &points, vector<vector<Point2f>> &trianglerPts) {
	int numTriangles = triangleVertices.size();
	trianglerPts.resize(numTriangles);
	for (int i = 0; i < numTriangles; i++) {
		vector<Point2f> triangle;
		for (int j = 0; j < 3; j++) {
			triangle.push_back(points[triangleVertices[i][j]]);
		}
		trianglerPts[i] = triangle;
	}
}

void paint_triangles(Mat &img, const vector<vector<Point2f>> &triangles) {
	int numTriangles = triangles.size();

	for (int i = 0; i < numTriangles; i++) {
		vector<Point> poly(3);

		for (int j = 0; j < 3; j++) {
			poly[j] = Point(cvRound(triangles[i][j].x), cvRound(triangles[i][j].y));
		}
		fillConvexPoly(img, poly, Scalar(i + 1));
	}
}

void solve_homography(const vector<Point2f> &srcPts1, const vector<Point2f> &srcPts2, Mat &homography) {
	assert(srcPts1.size() == srcPts2.size());
	homography = points_to_homogenous_mat(srcPts2) * points_to_homogenous_mat(srcPts1).inv();
}

void solve_homography(const vector<vector<Point2f>> &srcPts1,
		const vector<vector<Point2f>> &srcPts2,
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

void draw_contour_map(vector<vector<vector<Point>>> &collected, vector<Vec4i> &hierarchy, Mat &dst, int cols, int rows, int type) {
	dst = Mat::zeros(rows, cols, type);
	size_t cnt = 0;
	for (size_t i = 0; i < collected.size(); ++i) {
		auto &contours = collected[i];
		double shade = 0;

		cnt += contours.size();
		cerr << i << "/" << (collected.size() - 1) << '\r';
		for (size_t j = 0; j < contours.size(); ++j) {
			shade = 32.0 + 223.0 * (double(j) / contours.size());
			drawContours(dst, contours, j, { shade, shade, shade }, 1.0, LINE_4, hierarchy, 0);
		}
	}
	cerr << endl;
}

void adjust_contrast_and_brightness(const Mat& src, Mat& dst, double contrast, double lowcut) {
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
	dst.convertTo(hc, -1, contrast , brightness);
	hc.copyTo(dst);
}

double feature_metric(const Mat &grey1) {
	Mat corners;
	cornerHarris(grey1, corners, 2,3,0.04);
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

	if(cnz2 < (cnz1 * 0.9)) {
		int i = 0;
		while (cnz2 < (cnz1 * 0.9)) {
			Mat blurred1;
			GaussianBlur(decimated1, blurred1, {i * 2 + 1,i * 2 + 1}, i);
			decimated1 = blurred1.clone();
			cvtColor(decimated1, grey1, COLOR_RGB2GRAY);
			cnz1 = feature_metric(grey1);
			cerr << "redecimate features 1: " << cnz1 << "\r";
			++i;
		}
		cerr << endl;
	} else if(cnz1 < (cnz2 * 0.9)) {
		int i = 0;
		while (cnz1 < (cnz2 * 0.9)) {
			Mat blurred2;
			GaussianBlur(decimated2, blurred2, {i * 2 + 1,i * 2 + 1}, i);
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
	}
}

void extract_contours(const Mat &img1, const Mat &img2, Mat &allContours1, Mat &allContours2, vector<vector<vector<Point>>> &collected1, vector<vector<vector<Point>>> &collected2, vector<Mat> &contourLayers1, vector<Mat> &contourLayers2) {
	Mat grey1, grey2;
	cvtColor(img1, grey1, COLOR_RGB2GRAY);
	cvtColor(img2, grey2, COLOR_RGB2GRAY);

	double localSensitivity = 1;
	double t1 = 0;
	double t2 = 255;
	double highLimit = countNonZero(grey1) * 0.75;
	Mat thresh1, thresh2;
	vector<Vec4i> hierarchy1;
	vector<vector<Point>> contours1;
	size_t numContours = 0;
	threshold(grey1, thresh1, t1, t2, 0);
	size_t cnt = 0;
	do {
		t1 = localSensitivity;
		t2 = 255;
		if (t1 >= 255)
			t1 = 255;

		if (t2 >= 255)
			t2 = 255;
		threshold(grey1, thresh1, t1, t2, 0);
		cnt = countNonZero(thresh1);
		if (cnt > highLimit)
			++localSensitivity;
	} while (cnt > highLimit);

	t1 = 0;
	t2 = 0;
	localSensitivity = localSensitivity / 255.0;
	cerr << "thresholding 1" << endl;
	for (off_t i = 0; i < 16; ++i) {
		cerr << i << "/15" << '\r';
		t1 = max(0, min(255, (int) round(localSensitivity + (i * 16.0 * Settings::instance().contour_sensitivity))));
		t2 = max(0, min(255, (int) round(localSensitivity + ((i + 1) * 16.0 * Settings::instance().contour_sensitivity))));
		threshold(grey1, thresh1, t1, t2, 0);
		findContours(thresh1, contours1, hierarchy1, RETR_TREE, CHAIN_APPROX_TC89_KCOS);
		collected1.push_back(contours1);
		numContours += contours1.size();
		contours1.clear();
	}
	cerr << endl;

	Mat cmap1, cmap2;
	cerr << "draw map 1: " << numContours << endl;
	numContours = 0;
	draw_contour_map(collected1, hierarchy1, cmap1, grey1.cols, grey1.rows, grey1.type());

	cerr << "thresholding 2" << endl;
	vector<vector<Point>> contours2;
	vector<Vec4i> hierarchy2;
	for (off_t j = 0; j < 16; ++j) {
		cerr << j << "/15" << '\r';
		t1 = min(255, (int) round(localSensitivity + (j * 16 * Settings::instance().contour_sensitivity)));
		t2 = min(255, (int) round(localSensitivity + ((j + 1) * 16 * Settings::instance().contour_sensitivity)));
		threshold(grey2, thresh2, t1, t2, 0);
		findContours(thresh2, contours2, hierarchy2, RETR_TREE, CHAIN_APPROX_TC89_KCOS);
		collected2.push_back(contours2);
		numContours += contours2.size();
		contours2.clear();
	}
	cerr << endl;

	cerr << "draw map 2: " << numContours << endl;
	draw_contour_map(collected2, hierarchy2, cmap2, grey2.cols, grey2.rows, grey2.type());

	allContours1 = cmap1.clone();
	allContours2 = cmap2.clone();
	show_image("cmap1", cmap1);
	show_image("cmap2", cmap2);

	contourLayers1.clear();
	contourLayers2.clear();
	size_t minC = std::min(collected1.size(), collected2.size());
	contourLayers1.resize(minC);
	contourLayers2.resize(minC);

	cerr << "draw all contours -> " << endl;
	for (size_t i = 0; i < 1; ++i) {
		cerr << i << "/" << (minC - 1) << '\r';
		Mat &cont1 = contourLayers1[i];
		Mat &cont2 = contourLayers2[i];
		cont1 = Mat::zeros(grey1.rows, grey1.cols, grey1.type());
		cont2 = Mat::zeros(grey2.rows, grey2.cols, grey2.type());
		contours1 = collected1[i];
		contours2 = collected2[i];

		for (size_t j = 0; j < contours1.size(); ++j) {
			cv::drawContours(cont1, contours1, j, { 255 }, 1.0, cv::LINE_4, hierarchy1, 0);
		}

		for (size_t j = 0; j < contours2.size(); ++j) {
			cv::drawContours(cont2, contours2, j, { 255 }, 1.0, cv::LINE_4, hierarchy2, 0);
		}

//		cvtColor(cont1, cont1, cv::COLOR_RGB2GRAY);
//		cvtColor(cont2, cont2, cv::COLOR_RGB2GRAY);
	}
	cerr << endl;
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

	//create a radial mask to bias the contrast towards the center
	Mat radial = Mat::ones(grey1.rows, grey1.cols, CV_32F);
	draw_radial_gradiant(radial);

	//convert the images and masks to floating point for the subsequent multiplication
	Mat grey1Float, grey2Float, fgMask1Float, fgMask2Float, radialMaskFloat;
	grey1.convertTo(grey1Float, CV_32F, 1.0 / 255.0);
	grey2.convertTo(grey2Float, CV_32F, 1.0 / 255.0);
	fgMask1.convertTo(fgMask1Float, CV_32F, 1.0 / 255.0);
	fgMask2.convertTo(fgMask2Float, CV_32F, 1.0 / 255.0);
	radial.convertTo(radialMaskFloat, CV_32F, 1.0 / 255.0);

	//multiply the fg mask with the radial mask to emphasize features in the center of the image
	Mat finalMask1Float, finalMask2Float;
	if(Settings::instance().enable_radial_mask) {
		multiply(fgMask1Float, radialMaskFloat, finalMask1Float);
		multiply(fgMask2Float, radialMaskFloat, finalMask2Float);
	} else {
		fgMask1Float.copyTo(finalMask1Float);
		fgMask2Float.copyTo(finalMask2Float);
	}
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

	Mat blurred2Float, blurredMask2Float, maskedBlur2Float;
	GaussianBlur(grey2Float, blurred2Float, Size(23, 23), 3);
	GaussianBlur(finalMask2Float, blurredMask2Float, Size(23, 23), 3);
	maskedBlur2Float = blurred2Float / blurredMask2Float;
	addWeighted(grey2Float, 1.1, maskedBlur2Float, -0.1, 0, masked2);

	//convert back to 8-bit grey scale
	masked1.convertTo(masked1, CV_8U, 255.0);
	masked2.convertTo(masked2, CV_8U, 255.0);

	//adjust contrast and brightness
	adjust_contrast_and_brightness(masked1, foreground1, 2, 5);
	adjust_contrast_and_brightness(masked2, foreground2, 2, 5);

	show_image("fg1", foreground1);
	show_image("fg2", foreground2);
}

pair<double, Point2f> get_orientation(const vector<Point> &pts)
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
	Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)), static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
	//Store the eigenvalues and eigenvectors
	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; i++) {
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0), pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
	}

	return {atan2(eigen_vecs[0].y, eigen_vecs[0].x), cntr};
}

void translate(const Mat& src, Mat& dst, int x, int y) {
	float warpValues[] = { 1.0, 0.0, float(x), 0.0, 1.0, float(y) };
	Mat translation_matrix = Mat(2, 3, CV_32F, warpValues);
	warpAffine(src, dst, translation_matrix, src.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
}

void rotate(const Mat& src, Mat& dst, Point2f center, double angle) {
	Mat rm = getRotationMatrix2D(center, angle, 1.0);
	warpAffine(src, dst, rm, src.size());
}

Point2f rotate_point(const cv::Point2f& inPoint, const double& angDeg) {
	double rad = angDeg * M_PI/180.0;
    cv::Point2f outPoint;
    outPoint.x = std::cos(rad)*inPoint.x - std::sin(rad)*inPoint.y;
    outPoint.y = std::sin(rad)*inPoint.x + std::cos(rad)*inPoint.y;
    return outPoint;
}

Point2f rotate_point(const cv::Point2f& inPoint, const cv::Point2f& center, const double& angDeg) {
    return rotate_point(inPoint - center, angDeg) + center;
}

void rotate_points(vector<Point2f>& pts, const Point2f& center, const double& angDeg) {
	for(auto& pt : pts) {
		pt = rotate_point(pt - center, angDeg) + center;
	}
}

void correct_alignment(const Mat &src1, const Mat &src2, Mat &dst1, Mat &dst2, vector<vector<vector<Point>>> &collected1, vector<vector<vector<Point>>> &collected2) {
	cerr << "correcting alignment" << endl;
	assert(!collected1.empty() && !collected1[0].empty()
			&& !collected2.empty() && !collected2[0].empty());

	vector<Point> flat1;
	vector<Point> flat2;
	for (auto &c : collected1) {
		for (auto &v : c) {
			if (c.size() > 1) {
				RotatedRect rr = minAreaRect(v);
				if (rr.size.width > src1.cols * 0.5 || rr.size.height > src1.rows * 0.5) {
					continue;
				}
			}
			for (auto &pt : v) {
				flat1.push_back(pt);
			}
		}
	}

	if (flat1.empty())
		flat1 = collected1[0][0];

	for (auto &c : collected2) {
		for (auto &v : c) {
			if (c.size() > 1) {
				RotatedRect rr = minAreaRect(v);
				if (rr.size.width > src2.cols * 0.5 || rr.size.height > src2.rows * 0.5) {
					continue;
				}
			}
			for (auto &pt : v) {
				flat2.push_back(pt);
			}
		}
	}

	if (flat2.empty())
		flat2 = collected2[0][0];

	RotatedRect rr1 = minAreaRect(flat1);
	RotatedRect rr2 = minAreaRect(flat2);
	auto o1 = get_orientation(flat1);
	auto o2 = get_orientation(flat2);
	double angle1, angle2;
	o1.first = o1.first * 180 / M_PI;
	o2.first = o2.first * 180 / M_PI;
	o1.first = o1.first < 0 ? o1.first + 360 : o1.first;
	o2.first = o2.first < 0 ? o2.first + 360 : o2.first;

	if (fabs(o1.first - rr1.angle) < 22.5) {
		angle1 = (o1.first + rr1.angle) / 2.0;
	} else {
		double drr = fabs(rr1.angle - rr2.angle);
		double dor = fabs(o1.first - o2.first);
		if (dor < drr) {
			angle1 = dor;
		} else {
			angle1 = drr;
		}
	}

	if (fabs(o2.first - rr2.angle) < 22.5) {
		angle2 = (o2.first + rr2.angle) / 2.0;
	} else {
		double drr = fabs(rr1.angle - rr2.angle);
		double dor = fabs(o1.first - o2.first);
		if (dor < drr) {
			angle2 = dor;
		} else {
			angle2 = drr;
		}
	}

	double targetAng = angle2 - angle1;
	Point2f center1 = rr1.center;
	Point2f center2 = rr2.center;
	Mat rotated2;
	rotate(src2, rotated2, center2, targetAng);
	translate(rotated2, dst2, center1.x - center2.x, center1.y - center2.y);
	dst1 = src1.clone();

//	Mat test, copy1, copy2;
//	copy1 = dst1.clone();
//	copy2 = dst2.clone();
//	double lastDistance = numeric_limits<double>::max();
//	map<double, Mat> morphDistMap;
//	Mat distanceMap = Mat::zeros(copy1.rows, copy1.cols, CV_32FC1);
//	Mat preview;
//
//	for(int x = 0; x < copy1.cols; ++x) {
//		for(int y = 0; y < copy1.rows; ++y) {
//			translate(copy2, test,  x,  y);
//			distanceMap.at<float>(y,x) = cheap_morph_distance(copy1, test);
//			cerr << ((double(x * copy1.rows + y) / double(copy1.cols * copy1.rows)) * 100.0) << "%" << "\n";
//			normalize(distanceMap, preview, 0, 1, NORM_MINMAX);
//			show_image("DM",1.0 - preview);
//			waitKey(1);
//		}
//	}
//	cerr << endl;
//
//	while(true)
//		waitKey(0);


//	Mat t, test, grey1, grey2;
//	cvtColor(dst1, grey1, COLOR_RGB2GRAY);
//	cvtColor(dst2, grey2, COLOR_RGB2GRAY);
//
//	double d1, d2, d3, d4;
//	double lastDistance = numeric_limits<double>::max();
//	map<double, Mat> morphDistMap;
//
//	t = dst2.clone();
//	for(size_t i = 0; i < 100; ++i) {
//		morphDistMap.clear();
//		translate(t, test,  1,  0);
//		d1 = cheap_morph_distance(dst1, test);
//		morphDistMap[d1] = test;
//		translate(t, test, -1,  0);
//		d2 = cheap_morph_distance(dst1, test);
//		morphDistMap[d2] = test;
//		translate(t, test,  0,  1);
//		d3 = cheap_morph_distance(dst1, test);
//		morphDistMap[d3] = test;
//		translate(t, test,  0, -1);
//		d4 = cheap_morph_distance(dst1, test);
//		morphDistMap[d4] = test;
//		const auto& p = *morphDistMap.begin();
//		cerr << "searching: " << p.first << "\n";
//		if(p.first < lastDistance) {
//			cerr << "found: " << p.first << "\n";
//			lastDistance = p.first;
//			dst2 = p.second.clone();
//		}
//		t = p.second.clone();
//	}
//	cerr << endl;
}

FaceDetector face;

void scale_points(vector<Point2f>& pts, double coef) {
	for(auto& pt : pts) {
		pt.x *= coef;
		pt.y *= coef;
	}
}

void scale_features(Features& ft, double coef) {
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

void find_matches(Mat &orig1, Mat &orig2, Mat &corrected1, Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, Mat &contourMap1, Mat &contourMap2) {
	vector<vector<vector<Point>>> collected1;
	vector<vector<vector<Point>>> collected2;
	vector<Mat> contourLayers1;
	vector<Mat> contourLayers2;
	Mat decimated1;
	Mat decimated2;

	Features ft1;
	Features ft2;

	if(Settings::instance().enable_face_detection) {
		ft1 = face.detect(orig1);
		ft2 = face.detect(orig2);
	}

	if(ft1.empty() || ft2.empty()) {
		cerr << "general algorithm..." << endl;
		Mat goodFeatures1, goodFeatures2;

		extract_foreground(orig1, orig2, goodFeatures1, goodFeatures2);
		decimate_features(orig1, orig2, decimated1, decimated2);
		extract_contours(decimated1, decimated2, contourMap1, contourMap2, collected1, collected2, contourLayers1, contourLayers2);
		if (Settings::instance().enable_auto_align) {
			correct_alignment(orig1, orig2, corrected1, corrected2, collected1, collected2);
			collected1.clear();
			collected2.clear();
			extract_foreground(corrected1, corrected2, goodFeatures1, goodFeatures2);
			decimate_features(corrected1, corrected2, decimated1, decimated2);
			extract_contours(decimated1, decimated2, contourMap1, contourMap2, collected1, collected2, contourLayers1, contourLayers2);
		} else {
			corrected1 = orig1.clone();
			corrected2 = orig2.clone();
		}

		assert(!contourLayers1.empty() && !contourLayers2.empty());
		assert(contourLayers1.size() == contourLayers2.size());

		show_image("gf1", goodFeatures1);
		show_image("gf2", goodFeatures2);

		Mat features1, features2;
		cerr << "find matches" << endl;
		for(size_t i = 0; i < contourLayers1.size(); ++i) {
			if(contourLayers1[i].empty() || contourLayers2[i].empty())
				continue;

			features1 = goodFeatures1 * 0.5 + contourLayers1[i] * 0.5;
			features2 = goodFeatures2 * 0.5 + contourLayers2[i] * 0.5;

			auto matches = find_keypoints(features1, features2);
			srcPoints1.insert(srcPoints1.end(), matches.first.begin(), matches.first.end());
			srcPoints2.insert(srcPoints2.end(), matches.second.begin(), matches.second.end());
		}
	} else {
		cerr << "face algorithm..." << endl;
		assert(!ft1.empty() && !ft2.empty());
		extract_contours(orig1, orig2, contourMap1, contourMap2, collected1, collected2, contourLayers1, contourLayers2);
		srcPoints1 = ft1.getAllPoints();
		srcPoints2 = ft2.getAllPoints();

		if (Settings::instance().enable_auto_align) {
			cerr << orig1.size() << "/" << orig2.size() << endl;

			double w1 = fabs(ft1.right_eye_[0].x - ft1.left_eye_[0].x);
			double w2 = fabs(ft2.right_eye_[0].x - ft2.left_eye_[0].x);
			double scale2 = w1 / w2;
			cerr << "scale: " << scale2 << endl;
			scale_features(ft2, scale2);
			Mat scaled2;
			resize(orig2, scaled2, Size{orig2.cols*scale2,orig2.rows*scale2});
			ft2 = face.detect(scaled2);

			Point2f eyeVec1 = ft1.right_eye_[0] - ft1.left_eye_[0];
			Point2f eyeVec2 = ft2.right_eye_[0] - ft2.left_eye_[0];
			Point2f center1(ft1.left_eye_[0].x + (eyeVec1.x / 2.0), ft1.left_eye_[0].y + (eyeVec1.y / 2.0));
			Point2f center2(ft2.left_eye_[0].x + (eyeVec2.x / 2.0), ft2.left_eye_[0].y + (eyeVec2.y / 2.0));
			double angle1 = atan2(eyeVec1.y, eyeVec1.x);
			double angle2 = atan2(eyeVec2.y, eyeVec2.x);
			double dy = center1.y - center2.y;
			double dx = center1.x - center2.x;

			Mat translated2;
			translate(scaled2, translated2, dx, dy);

			angle1 = angle1 * 180 / M_PI;
			angle2 = angle2 * 180 / M_PI;
			angle1 = angle1 < 0 ? angle1 + 360 : angle1;
			angle2 = angle2 < 0 ? angle2 + 360 : angle2;
			std::cerr << "angle: " << angle1 << "/" << angle2 << endl;
			double targetAng = angle2 - angle1;
			Mat rotated2;
			rotate(translated2, rotated2, center2, targetAng);
			rotate_points(srcPoints2, center2, -targetAng);
			cerr << rotated2.size() << "/" << orig2.size() << endl;

			double dw = rotated2.cols - orig2.cols;
			double dh = rotated2.rows - orig2.rows;
			corrected2 = rotated2(Rect(dw / 2, dh / 2, orig2.cols, orig2.rows));
			corrected1 = orig1.clone();
			assert(corrected1.cols == corrected2.cols && corrected1.rows == corrected2.rows);
		} else {
			corrected1 = orig1.clone();
			corrected2 = orig2.clone();
		}
	}

	cerr << "contour points: " << srcPoints1.size() << "/" << srcPoints2.size() << endl;
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

void match_points_by_proximity(vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, int cols, int rows) {
	multimap<double, pair<Point2f, Point2f>> distanceMap;
	std::random_shuffle(srcPoints1.begin(), srcPoints1.end());
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
	auto distribution = calculate_sum_mean_and_sd(distanceMap);
	assert(!distanceMap.empty());
	if(get<1>(distribution) == 0 || get<2>(distribution) == 0)
		return;
	srcPoints1.clear();
	srcPoints2.clear();
	double distance = (*distanceMap.rbegin()).first;
	double mean = get<1>(distribution);
	double sd = get<2>(distribution);
	double highZScore = (fabs(distance - mean) / sd) / (max(sd, mean) - fabs(sd - mean));
	assert(highZScore > 0);
	double zScore = 0;
	double value = 0;
	double limit = 0.035 * Settings::instance().match_tolerance * highZScore * fabs(sd - mean);

	for (auto it = distanceMap.rbegin(); it != distanceMap.rend(); ++it) {
		value = (*it).first;
		zScore = (mean / sd) - fabs((value - mean) / sd);

		if (value < mean && zScore < limit) {
			srcPoints1.push_back((*it).second.first);
			srcPoints2.push_back((*it).second.second);
		}
	}

	assert(srcPoints1.size() == srcPoints2.size());
	check_points(srcPoints1, cols, rows);
	check_points(srcPoints2, cols, rows);
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
	show_image("matches reduced", matMatches);

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
	vector<vector<Point2f>> triangleSrc1, triangleSrc2, triangleMorph;
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
	Mat_<Vec3f> lapBlend = lb.blend().clone();
	lapBlend.convertTo(dst, origImg1.depth(), 255.0);
	Mat analysis = dst.clone();
	Mat prev = last.clone();
	if (prev.empty())
		prev = dst.clone();
	draw_morph_analysis(dst, prev, analysis, SourceImgSize, subDiv1, subDiv2, subDivMorph, { 0, 0, 255 });
	show_image("mesh", analysis);
	return 0;
}
}
