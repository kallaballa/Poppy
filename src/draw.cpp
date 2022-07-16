#include "draw.hpp"
#include "util.hpp"

#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/ocl.hpp>

using std::cerr;
using std::cout;
using std::endl;

namespace poppy {
void plot(Mat &img, vector<Point2f> points, Scalar color, int radius) {
	for (Point2f p : points)
		circle(img, p, radius, color, radius * 2);
}

void draw_radial_gradiant(Mat &grad) {
	cv::Point center(grad.cols / 2.0, grad.rows / 2.0);
	cv::Point point;
	double maxDist = hypot(grad.cols / 2.0, grad.rows / 2.0);
	for (int row = 0; row < grad.rows; ++row) {
		for (int col = 0; col < grad.cols; ++col) {
			point.x = col;
			point.y = row;
			double dist = euclidean_distance(center, point) / maxDist;
//			grad.at<float>(row, col) = dist;
			grad.at<float>(row, col) = pow(sin(sin(dist * (M_PI/2)) * (M_PI/2)),12);
		}
	}

	cv::normalize(grad, grad, 0, 255, cv::NORM_MINMAX, CV_8U);
	cv::bitwise_not(grad, grad);
//	show_image("grad", grad);
}

Mat draw_radial_gradiant2(int width, int height) {
	Mat gradFloat = Mat::ones(height, width, CV_32F);
	Mat grad;
	cv::Point center(width / 2.0, height / 2.0);
	cv::Point point;
	double maxDist = hypot(width / 2.0, height / 2.0);
	for (int row = 0; row < height; ++row) {
		for (int col = 0; col < width; ++col) {
			point.x = col;
			point.y = row;
			double dist = euclidean_distance(center, point) / maxDist;
			gradFloat.at<float>(row, col) = pow(sin(sin(dist * (M_PI/2)) * (M_PI/2)),16);
		}
	}
	gradFloat.convertTo(grad, CV_8U, 255.0);
	cv::normalize(grad, grad, 0, 255, cv::NORM_MINMAX);
	cv::bitwise_not(grad, grad);
	grad.convertTo(gradFloat, CV_32F, 1.0/255.0);
	return gradFloat;
}


void draw_contour_map(Mat &dst, vector<Mat>& contourLayers, const vector<vector<vector<Point2f>>> &collected, const vector<Vec4i> &hierarchy, int cols, int rows, int type) {
	Mat map = Mat::zeros(rows, cols, type);
	for (size_t i = 0; i < collected.size(); ++i) {
		auto &contours = collected[i];
		double shade = (32.0 + 223.0 * (double(i) / collected.size()));
		cerr << i + 1 << "/" << collected.size() << '\r';
		vector<vector<Point>> tmp = convertContourFrom2f(contours);
		Mat layer = Mat::zeros(rows, cols, type);
		for (size_t j = 0; j < tmp.size(); ++j) {
			drawContours(layer, tmp, j, { 255 }, 1.0, LINE_4, hierarchy, 0);
			drawContours(map, tmp, j, { shade }, 1.0, LINE_4, hierarchy, 0);
		}
		contourLayers.push_back(layer);
	}

	dst = map.clone();
}

void draw_delaunay(Mat &dst, const Size &size, Subdiv2D &subdiv, Scalar delaunay_color) {
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
			line(dst, pt[0], pt[1], delaunay_color, 1, cv::LINE_4, 0);
			line(dst, pt[1], pt[2], delaunay_color, 1, cv::LINE_4, 0);
			line(dst, pt[2], pt[0], delaunay_color, 1, cv::LINE_4, 0);
		}
	}
}
#ifndef _WASM
void draw_flow_heightmap(const Mat &morphed, const Mat &last, Mat &dst) {
	UMat flowUmat(morphed.rows, morphed.cols, morphed.type());
	Mat flow;
	Mat grey1, grey2;
	cvtColor(last, grey1, cv::COLOR_RGB2GRAY);
	cvtColor(morphed, grey2, cv::COLOR_RGB2GRAY);
	calcOpticalFlowFarneback(grey1, grey2, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
	flowUmat.copyTo(flow);
	dst = morphed.clone();
	uint32_t color;

	double maxMag = 0;
	double mag = 0;
	for (off_t x = 0; x < morphed.cols; ++x) {
		for (off_t y = 0; y < morphed.rows; ++y) {
			const Point2f &flv1 = flow.at<Point2f>(y, x);
			mag = hypot(flv1.x, flv1.y);
			if (mag > maxMag)
				maxMag = mag;
		}
	}

	for (off_t x = 0; x < morphed.cols; ++x) {
		for (off_t y = 0; y < morphed.rows; ++y) {
//			circle(dst, Point(x, y), 1, Scalar(0), -1);
			const Point2f flv1 = flow.at<Point2f>(y, x);
			double mag = hypot(flv1.x, flv1.y) / maxMag;
			color = std::round(double(255) * (double) mag);
			circle(dst, Point(x, y), 1, Scalar(color, color, color), -1);
		}
	}
	show_image("fhm", dst);
}

void draw_flow_vectors(const Mat &morphed, const Mat &last, Mat &dst) {
	UMat flowUmat;
	Mat flow;
	Mat grey1, grey2;
	cvtColor(last, grey1, cv::COLOR_RGB2GRAY);
	cvtColor(morphed, grey2, cv::COLOR_RGB2GRAY);

	calcOpticalFlowFarneback(grey1, grey2, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
	flowUmat.copyTo(flow);
	dst = morphed.clone();

	double diag = hypot(morphed.cols, morphed.rows);
	for (off_t x = 0; x < morphed.cols; ++x) {
		for (off_t y = 0; y < morphed.rows; ++y) {
			const Point2f flv1 = flow.at<Point2f>(y, x) * 10;
			double len = hypot(flv1.x - x, flv1.y - y);
			if(len > 100) {
				cv::Mat3f bgr(cv::Vec3f(0,0,0));
				cv::Mat3f hsv(cv::Vec3f(double(179) * (double(len) / diag), 100 , 100));
				cvtColor(hsv, bgr, COLOR_HSV2BGR);
				line(dst, Point(x, y), Point(cvRound(x + flv1.x), cvRound(y + flv1.y)), Scalar(bgr.at<Vec3b>(0,0)[0],bgr.at<Vec3b>(0,0)[1],bgr.at<Vec3b>(0,0)[2]));
				circle(dst, Point(x, y), 1, Scalar(0, 0, 0), -1);
			}
		}
	}
	show_image("fv", dst);
}

void draw_flow_highlight(const Mat &morphed, const Mat &last, Mat &dst) {
	Mat flowv;
	Mat flowm;
	draw_flow_vectors(morphed, last, flowv);
	draw_flow_heightmap(morphed, last, flowm);
	Mat overlay;
	normalize(flowv.mul(flowm), overlay, 255.0, 0.0, NORM_MINMAX);
	dst = morphed * 0.7 + overlay * 0.3;
}
#endif
void draw_morph_analysis(const Mat &morphed, const Mat &last, Mat &dst, const Size &size, Subdiv2D &subdiv1, Subdiv2D &subdiv2, Subdiv2D &subdivMorph, Scalar delaunay_color) {
//	draw_flow_highlight(morphed, last, dst);
//	UMat flowUmat;
//	Mat flow;
//	Mat grey1, grey2;
//	cvtColor(last, grey1, cv::COLOR_RGB2GRAY);
//	cvtColor(morphed, grey2, cv::COLOR_RGB2GRAY);
//
//	calcOpticalFlowFarneback(grey1, grey2, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
//	flowUmat.copyTo(flow);
//	cv::Mat xy[2]; //X,Y
//	cv::split(flow, xy);
//	cv::Mat magnitude, angle;
//	cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);
//	double mag_max;
//	cv::minMaxLoc(magnitude, 0, &mag_max);
//	magnitude.convertTo(magnitude, -1, 1.0 / mag_max);
//	cv::Mat _hsv[3], hsv;
//	_hsv[0] = angle;
//	_hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
//	_hsv[2] = magnitude;
//	cv::merge(_hsv, 3, hsv);
//	cv::Mat bgr;//CV_32FC3 matrix
//	cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
//
//	show_image("flow", bgr);

	draw_delaunay(dst, size, subdiv1, { 255, 0, 0 });
	draw_delaunay(dst, size, subdiv2, { 0, 255, 0 });
	draw_delaunay(dst, size, subdivMorph, { 0, 0, 255 });
}

void draw_matches(const Mat &grey1, const Mat &grey2, Mat &dst, std::vector<Point2f> &ptv1, std::vector<Point2f> &ptv2) {
	Mat images = cv::Mat::zeros( { grey1.cols * 2, grey1.rows }, CV_8UC1);
	Mat lines = cv::Mat::ones( { grey1.cols * 2, grey1.rows }, CV_8UC1);

	grey1.copyTo(images(cv::Rect(0, 0, grey1.cols, grey1.rows)));
	grey2.copyTo(images(cv::Rect(grey1.cols, 0, grey1.cols, grey1.rows)));

	for (size_t i = 0; i < std::min(ptv1.size(), ptv2.size()); ++i) {
		Point2f pt2 = ptv2[i];
		pt2.x += grey1.cols;
		line(lines, ptv1[i], pt2, { 127 }, 1, cv::LINE_4, 0);
	}

	Mat result = images * 0.5 + lines * 0.5;
	cvtColor(result, dst, COLOR_GRAY2RGB);
}

void draw_matches(const Mat &grey1, const Mat &grey2, Mat &dst, std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2) {
	Mat images = cv::Mat::zeros( { grey1.cols * 2, grey1.rows }, CV_8UC1);
	Mat lines = cv::Mat::ones( { grey1.cols * 2, grey1.rows }, CV_8UC1);

	grey1.copyTo(images(cv::Rect(0, 0, grey1.cols, grey1.rows)));
	grey2.copyTo(images(cv::Rect(grey1.cols, 0, grey1.cols, grey1.rows)));

	for (size_t i = 0; i < kpv1.size(); ++i) {
		Point2f pt2 = kpv2[i].pt;
		pt2.x += grey1.cols;
		line(lines, kpv1[i].pt, pt2, { 127 }, 1, cv::LINE_4, 0);
	}

	images += 1;
	Mat result = images.mul(lines);
	cvtColor(result, dst, COLOR_GRAY2RGB);
}

}
