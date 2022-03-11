#include "draw.hpp"
#include "util.hpp"

#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/ocl.hpp>

namespace poppy {
double euclidean_distance(cv::Point center, cv::Point point) {
	double distance = std::sqrt(
			std::pow(center.x - point.x, 2) + std::pow(center.y - point.y, 2));
	return distance;
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
			grad.at<float>(row, col) = pow(sin(sin(sin(dist) * (M_PI/2))* (M_PI/2)),8);
		}
	}

	cv::normalize(grad, grad, 0, 255, cv::NORM_MINMAX, CV_8U);
	cv::bitwise_not(grad, grad);
	show_image("grad", grad);
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

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
				{
			line(dst, pt[0], pt[1], delaunay_color, 1, cv::LINE_AA, 0);
			line(dst, pt[1], pt[2], delaunay_color, 1, cv::LINE_AA, 0);
			line(dst, pt[2], pt[0], delaunay_color, 1, cv::LINE_AA, 0);
		}
	}
}
#ifndef _WASM
void draw_flow_heightmap(const Mat &morphedGrey, const Mat &lastGrey, Mat &dst) {
	UMat flowUmat;
	Mat flow;
	Mat grey1 = morphedGrey.clone();
	Mat grey2 = lastGrey.clone();

	calcOpticalFlowFarneback(grey1, grey2, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
	flowUmat.copyTo(flow);
	dst = morphedGrey.clone();
	uint32_t color;

	double maxMag = 0;
	double mag = 0;
	for (off_t x = 0; x < morphedGrey.cols; ++x) {
		for (off_t y = 0; y < morphedGrey.rows; ++y) {
			const Point2f &flv1 = flow.at<Point2f>(y, x);
			mag = hypot(flv1.x, flv1.y);
			if (mag > maxMag)
				maxMag = mag;
		}
	}

	for (off_t x = 0; x < morphedGrey.cols; ++x) {
		for (off_t y = 0; y < morphedGrey.rows; ++y) {
//			circle(dst, Point(x, y), 1, Scalar(0), -1);
			const Point2f flv1 = flow.at<Point2f>(y, x);
			double mag = hypot(flv1.x, flv1.y) / maxMag;
			color = std::round(double(255) * (double) mag);
			circle(dst, Point(x, y), 1, Scalar(color, color, color), -1);
		}
	}
//	show_image("fhm", dst);
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
	uint32_t color;

	double diag = hypot(morphed.cols, morphed.rows);
	for (off_t x = 0; x < morphed.cols; ++x) {
		for (off_t y = 0; y < morphed.rows; ++y) {
			const Point2f flv1 = flow.at<Point2f>(y, x) * 10;
			double len = hypot(flv1.x - x, flv1.y - y);
			color = std::round(double(255) * (double(len) / diag));
			line(dst, Point(x, y), Point(cvRound(x + flv1.x), cvRound(y + flv1.y)), Scalar(color, color, color));
			circle(dst, Point(x, y), 1, Scalar(0, 0, 0), -1);
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
		line(lines, ptv1[i], pt2, { 127 }, 1, cv::LINE_AA, 0);
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
		line(lines, kpv1[i].pt, pt2, { 127 }, 1, cv::LINE_AA, 0);
	}

	images += 1;
	Mat result = images.mul(lines);
	cvtColor(result, dst, COLOR_GRAY2RGB);
}

}
