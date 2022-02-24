#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <algorithm>

#include <boost/program_options.hpp>

#include <CGAL/Cartesian.h>
#include <CGAL/MP_Float.h>
#include <CGAL/Quotient.h>
#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Sweep_line_2_algorithms.h>

#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

typedef CGAL::Quotient<CGAL::MP_Float>                  NT;
typedef CGAL::Cartesian<NT>                             Kernel;
typedef Kernel::Point_2                                 Point_2;
typedef CGAL::Arr_segment_traits_2<Kernel>              Traits_2;
typedef Traits_2::Curve_2                               Segment_2;

double number_of_frames = 60;
double max_len_deviation = 20;
double max_ang_deviation = 0.3;
double max_pair_len_divider = 10;
double max_chop_len = 20;
double contour_sensitivity = 0.3;

using namespace cv;
using std::vector;
using std::chrono::microseconds;

namespace po = boost::program_options;

typedef unsigned char sample_t;

int ratioTest(std::vector<std::vector<cv::DMatch> >
		&matches) {
#ifndef _NO_OPENCV
	int removed = 0;
	// for all matches
	for (std::vector<std::vector<cv::DMatch> >::iterator
	matchIterator = matches.begin();
			matchIterator != matches.end(); ++matchIterator) {
		// if 2 NN has been identified
		if (matchIterator->size() > 1) {
			// check distance ratio
			if ((*matchIterator)[0].distance /
					(*matchIterator)[1].distance > 0.7) {
				matchIterator->clear(); // remove match
				removed++;
			}
		} else { // does not have 2 neighbours
			matchIterator->clear(); // remove match
			removed++;
		}
	}
	return removed;
#else
        return 0;
#endif
}

cv::Mat ransacTest(
		const std::vector<cv::DMatch> &matches,
		const std::vector<cv::KeyPoint> &keypoints1,
		const std::vector<cv::KeyPoint> &keypoints2,
		std::vector<cv::DMatch> &outMatches)
		{
#ifndef _NO_OPENCV
	// Convert keypoints into Point2f
	std::vector<cv::Point2f> points1, points2;
	for (std::vector<cv::DMatch>::
	const_iterator it = matches.begin();
			it != matches.end(); ++it) {
		// Get the position of left keypoints
		float x = keypoints1[it->queryIdx].pt.x;
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x, y));
		// Get the position of right keypoints
		x = keypoints2[it->trainIdx].pt.x;
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x, y));
	}
	// Compute F matrix using RANSAC
	std::vector<uchar> inliers(points1.size(), 0);
	std::vector<cv::Point2f> out;
	//cv::Mat fundemental= cv::findFundamentalMat(points1, points2, out, CV_FM_RANSAC, 3, 0.99);

	cv::Mat fundemental = findFundamentalMat(
			cv::Mat(points1), cv::Mat(points2), // matching points
			inliers,      // match status (inlier or outlier)
			cv::FM_RANSAC, // RANSAC method
			3.0,     // distance to epipolar line
			0.99);  // confidence probability

	// extract the surviving (inliers) matches
	std::vector<uchar>::const_iterator
	itIn = inliers.begin();
	std::vector<cv::DMatch>::const_iterator
	itM = matches.begin();
	// for all matches
	for (; itIn != inliers.end(); ++itIn, ++itM) {
		if (*itIn) { // it is a valid match
			outMatches.push_back(*itM);
		}
	}
	return fundemental;
#else
    return cv::Mat();
#endif
}

void symmetryTest(
		const std::vector<std::vector<cv::DMatch>> &matches1,
		const std::vector<std::vector<cv::DMatch>> &matches2,
		std::vector<cv::DMatch> &symMatches) {
#ifndef _NO_OPENCV
	// for all matches image 1 -> image 2
	for (std::vector<std::vector<cv::DMatch>>::
	const_iterator matchIterator1 = matches1.begin();
			matchIterator1 != matches1.end(); ++matchIterator1) {
		// ignore deleted matches
		if (matchIterator1->size() < 2)
			continue;
		// for all matches image 2 -> image 1
		for (std::vector<std::vector<cv::DMatch>>::
		const_iterator matchIterator2 = matches2.begin();
				matchIterator2 != matches2.end();
				++matchIterator2) {
			// ignore deleted matches
			if (matchIterator2->size() < 2)
				continue;
			// Match symmetry test
			if ((*matchIterator1)[0].queryIdx ==
					(*matchIterator2)[0].trainIdx &&
					(*matchIterator2)[0].queryIdx ==
							(*matchIterator1)[0].trainIdx) {
				// add symmetrical match
				symMatches.push_back(
						cv::DMatch((*matchIterator1)[0].queryIdx,
								(*matchIterator1)[0].trainIdx,
								(*matchIterator1)[0].distance));
				break; // next match in image 1 -> image 2
			}
		}
	}
#endif
}

struct LessPointOp {
	bool operator()(const Point2f &lhs, const Point2f &rhs) const {
		return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
	}
};

bool operator==(const KeyPoint &kp1, const KeyPoint &kp2) {
	return kp1.pt.x == kp2.pt.x && kp1.pt.y == kp2.pt.y;
}

double distance(const Point2f &p1, const Point2f &p2) {
	return hypot(p2.x - p1.x, p2.y - p1.y);
}

void canny_threshold(const Mat &src, Mat &detected_edges, double thresh) {
	detected_edges = src.clone();
	GaussianBlur(src, detected_edges, Size(9, 9), 1);
	/// Canny detector
	Canny(detected_edges, detected_edges, thresh, thresh * 2);
}

void angle_test(std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols) {
	double maxDeviationPercent = max_ang_deviation;
	double avg = 0;
	double total = 0;

	for (size_t i = 0; i < kpv1.size(); ++i) {
		total += M_PI + std::atan2(kpv2[i].pt.y - kpv1[i].pt.y, (cols + kpv2[i].pt.x) - kpv1[i].pt.x);
	}

	avg = total / kpv1.size();
	double dev = avg / (100 / maxDeviationPercent);
	double min = avg - (dev / 2);
	double max = min + dev;

	std::vector<KeyPoint> new1;
	std::vector<KeyPoint> new2;

	for (size_t i = 0; i < kpv1.size(); ++i) {
		double angle = M_PI + std::atan2(kpv2[i].pt.y - kpv1[i].pt.y, (cols + kpv2[i].pt.x) - kpv1[i].pt.x);

		if (angle > min && angle < max) {
			new1.push_back(kpv1[i]);
			new2.push_back(kpv2[i]);
		}
	}
	kpv1 = new1;
	kpv2 = new2;
}

void angle_test(std::vector<Point2f> &ptv1, std::vector<Point2f> &ptv2, int cols) {
	std::vector<KeyPoint> kpv1;
	std::vector<KeyPoint> kpv2;

	for (auto pt : ptv1) {
		kpv1.push_back( { pt, 1 });
	}

	for (auto pt : ptv2) {
		kpv2.push_back( { pt, 1 });
	}

	angle_test(kpv1, kpv2, cols);

	ptv1.clear();
	ptv2.clear();

	for (auto kp : kpv1) {
		ptv1.push_back(kp.pt);
	}

	for (auto kp : kpv2) {
		ptv2.push_back(kp.pt);
	}
}
void length_test(std::vector<std::tuple<KeyPoint, KeyPoint, double>> edges, std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols) {
	double maxDeviationPercent = max_len_deviation;
	double avg = 0;
	double total = 0;

	for (auto e : edges) {
		total += std::get<2>(e);
	}

	avg = total / edges.size();
	double dev = avg / (100 / maxDeviationPercent);
	double min = avg - (dev / 2);
	double max = min + dev;

	kpv1.clear();
	kpv2.clear();

	for (auto e : edges) {
		double len = std::get<2>(e);
		if (len > min && len < max) {
			kpv1.push_back(std::get<0>(e));
			kpv2.push_back(std::get<1>(e));
		}
	}
}

void length_test(std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols) {
	std::vector<std::tuple<KeyPoint, KeyPoint, double>> edges;
	edges.reserve(10000);
	Point2f p1, p2;
	for (auto &kp1 : kpv1) {
		for (auto &kp2 : kpv2) {
			edges.push_back( { kp1, kp2, distance(kp1.pt, Point2f(kp2.pt.x + cols, kp2.pt.y)) });
		}
	}

	length_test(edges, kpv1, kpv2, cols);
}

void length_test(std::vector<Point2f> &ptv1, std::vector<Point2f> &ptv2, int cols) {
	std::vector<KeyPoint> kpv1;
	std::vector<KeyPoint> kpv2;

	for (auto pt : ptv1) {
		kpv1.push_back( { pt, 1 });
	}

	for (auto pt : ptv2) {
		kpv2.push_back( { pt, 1 });
	}

	length_test(kpv1, kpv2, cols);

	ptv1.clear();
	ptv2.clear();

	for (auto kp : kpv1) {
		ptv1.push_back(kp.pt);
	}

	for (auto kp : kpv2) {
		ptv2.push_back(kp.pt);
	}
}

void make_delaunay_mesh(const Size &size, Subdiv2D &subdiv, std::vector<Point2f> &dstPoints) {
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);
	Rect rect(0, 0, size.width, size.height);

	for (size_t i = 0; i < triangleList.size(); i++) {
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

		//Draw triangles completely inside the image.
		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
			dstPoints.push_back(pt[0]);
			dstPoints.push_back(pt[1]);
			dstPoints.push_back(pt[2]);
		}
	}
}

// Draw delaunay triangles
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

		//Draw triangles completely inside the image.
		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
				{
			line(dst, pt[0], pt[1], delaunay_color, 1, cv::LINE_AA, 0);
			line(dst, pt[1], pt[2], delaunay_color, 1, cv::LINE_AA, 0);
			line(dst, pt[2], pt[0], delaunay_color, 1, cv::LINE_AA, 0);
		}
	}
}

void draw_flow_heightmap(const Mat &morphed, const Mat &last, Mat &dst) {
	UMat flowUmat;
	Mat flow;
	Mat grey1, grey2;
	cvtColor(last, grey1, cv::COLOR_RGB2GRAY);
	cvtColor(morphed, grey2, cv::COLOR_RGB2GRAY);

	calcOpticalFlowFarneback(grey1, grey2, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
	flowUmat.copyTo(flow);
	dst = morphed.clone();
	uint32_t color;

	for (off_t x = 0; x < morphed.cols; ++x) {
		for (off_t y = 0; y < morphed.rows; ++y) {
			circle(dst, Point(x, y), 1, Scalar(0), -1);
			const Point2f flv1 = flow.at<Point2f>(y, x);
			double mag = hypot(flv1.x, flv1.y);
			color = std::round(double(255) * (double) mag);
			circle(dst, Point(x, y), 1, Scalar(color), -1);
		}
	}
	imshow("fhm", dst);
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
			line(dst, Point(x, y), Point(cvRound(x + flv1.x), cvRound(y + flv1.y)), Scalar(color));
			circle(dst, Point(x, y), 1, Scalar(0, 0, 0), -1);
		}
	}
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

void collect_flow_centers(const Mat& morphed, const Mat& last, std::vector<std::pair<Point2f,double>>& highlightCenters) {
	Mat flowm;
	Mat grey;
	draw_flow_heightmap(morphed, last, flowm);
	cvtColor(flowm, grey, cv::COLOR_RGB2GRAY);

	Mat overlay;
	Mat thresh;
	normalize(grey, overlay, 255.0, 0.0, NORM_MINMAX);
	GaussianBlur(overlay, overlay, Size(13, 13), 2);
	cv::threshold(overlay, thresh, 254, 255, 0);
	std::vector<std::vector<cv::Point>> contours;
	vector<Vec4i> hierarchy;
	cv::findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);
	Rect rect(0, 0, morphed.cols, morphed.rows);

	for (auto &ct : contours) {
		auto br = boundingRect(ct);
		double cx = br.x + br.width / 2.0;
		double cy = br.y + br.height / 2.0;
		Point2f pt(cx, cy);
		if (rect.contains(pt)) {
			highlightCenters.push_back({pt, hypot(br.width, br.height)});
		}
	}
}



void draw_morph_analysis(const Mat &morphed, const Mat &last, Mat &dst, const Size &size, Subdiv2D &subdiv1, Subdiv2D &subdiv2, Subdiv2D &subdivMorph, Scalar delaunay_color) {
//	std::vector<std::pair<Point2f,double>> highlights;
//	collect_flow_centers(morphed, last, highlights);
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

//	vector<Vec6f> triangleList;
//	subdivMorph.getTriangleList(triangleList);
//	vector<Point> pt(3);
//	Rect rect(0, 0, size.width, size.height);
//
//	for (size_t i = 0; i < triangleList.size(); i++) {
//		Vec6f t = triangleList[i];
//		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
//		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
//		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
//
//		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
//			const Point2f flow1 = flow.at<Point2f>(pt[0].y, pt[0].x) * 20;
//			const Point2f flow2 = flow.at<Point2f>(pt[1].y, pt[1].x) * 20;
//			const Point2f flow3 = flow.at<Point2f>(pt[2].y, pt[2].x) * 20;
//			line(dst, pt[0], Point(cvRound(pt[0].x + flow1.x), cvRound(pt[0].y + flow1.y)), Scalar(255, 255, 0));
//			line(dst, pt[1], Point(cvRound(pt[1].x + flow2.x), cvRound(pt[1].y + flow2.y)), Scalar(255, 255, 0));
//			line(dst, pt[2], Point(cvRound(pt[2].x + flow3.x), cvRound(pt[2].y + flow3.y)), Scalar(255, 255, 0));
//		}
//	}
//
//	for (auto &h : highlights) {
//		circle(dst, h.first, 1, Scalar(255, 255, 255), -1);
//	}
}

void draw_matches(const Mat &grey1, const Mat &grey2, Mat &dst, std::vector<Point2f> &ptv1, std::vector<Point2f> &ptv2) {
	Mat images = cv::Mat::zeros( { grey1.cols * 2, grey1.rows }, CV_8UC1);
	Mat lines = cv::Mat::ones( { grey1.cols * 2, grey1.rows }, CV_8UC1);

	grey1.copyTo(images(cv::Rect(0, 0, grey1.cols, grey1.rows)));
	grey2.copyTo(images(cv::Rect(grey1.cols, 0, grey1.cols, grey1.rows)));

	for (size_t i = 0; i < ptv1.size(); ++i) {
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

std::pair<std::vector<Point2f>, std::vector<Point2f>> find_matches(const Mat &grey1, const Mat &grey2) {
	cv::Ptr<cv::ORB> detector = cv::ORB::create(100);
	cv::Ptr<cv::ORB> extractor = cv::ORB::create();

	std::vector<KeyPoint> keypoints1, keypoints2;

	Mat descriptors1, descriptors2;
	detector->detect(grey1, keypoints1);
	detector->detect(grey2, keypoints2);

	detector->compute(grey1, keypoints1, descriptors1);
	detector->compute(grey2, keypoints2, descriptors2);

	Mat matMatches;

//	length_test(keypoints1, keypoints2, grey1.cols);
//	angle_test(keypoints1, keypoints2, grey1.cols);

	std::vector<Point2f> points1;
	std::vector<Point2f> points2;
	for (auto pt1 : keypoints1) {
		points1.push_back(pt1.pt);
	}

	for (auto pt2 : keypoints2) {
		points2.push_back(pt2.pt);
	}
	return {points1,points2};
}

std::pair<std::vector<Point2f>, std::vector<Point2f>> find_matches_classic(const Mat& img1, const Mat& img2, std::vector<std::pair<Point2f,double>>& high1, std::vector<std::pair<Point2f,double>>& high2) {
	Mat grey1, grey2;
	cvtColor(img1, grey1, cv::COLOR_RGB2GRAY);
	cvtColor(img2, grey2, cv::COLOR_RGB2GRAY);
	Mat b1 = Mat::zeros({grey1.cols, grey1.rows}, grey1.type());
	Mat b2 = b1.clone();

	for(auto h : high1) {
		circle(b1, h.first, h.second / 4.0, Scalar(255), 2);
	}

	for(auto h : high2) {
		circle(b2, h.first, h.second / 4.0, Scalar(255), 2);
	}

	cv::Ptr<cv::ORB> detector = cv::ORB::create(1000000);
	cv::Ptr<cv::ORB> extractor = cv::ORB::create();

	std::vector<KeyPoint> keypoints1, keypoints2;

	Mat descriptors1, descriptors2;
	detector->detect(b1, keypoints1);
	detector->detect(b2, keypoints2);

	detector->compute(b1, keypoints1, descriptors1);
	detector->compute(b2, keypoints2, descriptors2);

	cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::LshIndexParams(12, 20, 2);
	FlannBasedMatcher matcher1(indexParams);
	FlannBasedMatcher matcher2(indexParams);
	std::vector<std::vector<cv::DMatch>> matches1, matches2;
	std::vector<DMatch> goodMatches;
	std::vector<DMatch> symMatches;

	matcher1.knnMatch(descriptors1, descriptors2, matches1, 2);
	matcher2.knnMatch(descriptors2, descriptors1, matches2, 2);

	ratioTest(matches1);
	ratioTest(matches2);
	symmetryTest(matches1, matches2, symMatches);

	if (symMatches.empty()) {
		assert(false);
	}
	cv::Mat fundamental = ransacTest(symMatches, keypoints1, keypoints2, goodMatches);

	if (goodMatches.empty()) {
		assert(false);
	}

	std::vector<Point2f> points1;
	std::vector<Point2f> points2;

	for(auto& gm : goodMatches) {
		points1.push_back(keypoints1[gm.queryIdx].pt);
		points2.push_back(keypoints2[gm.trainIdx].pt);
	}

	return {points1,points2};
}

cv::Mat points_to_homogenous_mat(const std::vector<cv::Point2f> &pts) {
	int numPts = pts.size();
	cv::Mat homMat(3, numPts, CV_32FC1);
	for (int i = 0; i < numPts; i++) {
		homMat.at<float>(0, i) = pts[i].x;
		homMat.at<float>(1, i) = pts[i].y;
		homMat.at<float>(2, i) = 1.0;
	}
	return homMat;
}

void morph_points(std::vector<cv::Point2f> &srcPts1, std::vector<cv::Point2f> &srcPts2, std::vector<cv::Point2f> &dstPts, float s = 0.5) {
	assert(srcPts1.size() == srcPts2.size());

	int numPts = srcPts1.size();

	dstPts.resize(numPts);
	for (int i = 0; i < numPts; i++) {
		dstPts[i].x = (1.0 - s) * srcPts1[i].x + s * srcPts2[i].x;
		dstPts[i].y = (1.0 - s) * srcPts1[i].y + s * srcPts2[i].y;
	}
}

void get_triangle_indices(const cv::Subdiv2D &subDiv, const std::vector<cv::Point2f> &points, std::vector<cv::Vec3i> &triangleVertices) {
	std::vector<cv::Vec6f> triangles;
	subDiv.getTriangleList(triangles);

	int numTriangles = triangles.size();
	triangleVertices.clear();
	triangleVertices.reserve(numTriangles);
	for (int i = 0; i < numTriangles; i++) {
		std::vector<cv::Point2f>::const_iterator vert1, vert2, vert3;
		vert1 = std::find(points.begin(), points.end(), cv::Point2f(triangles[i][0], triangles[i][1]));
		vert2 = std::find(points.begin(), points.end(), cv::Point2f(triangles[i][2], triangles[i][3]));
		vert3 = std::find(points.begin(), points.end(), cv::Point2f(triangles[i][4], triangles[i][5]));

		cv::Vec3i vertex;
		if (vert1 != points.end() && vert2 != points.end() && vert3 != points.end()) {
			vertex[0] = vert1 - points.begin();
			vertex[1] = vert2 - points.begin();
			vertex[2] = vert3 - points.begin();
			triangleVertices.push_back(vertex);
		}
	}
}

void make_triangler_points(const std::vector<cv::Vec3i> &triangleVertices, const std::vector<cv::Point2f> &points, std::vector<std::vector<cv::Point2f>> &trianglerPts) {
	int numTriangles = triangleVertices.size();
	trianglerPts.resize(numTriangles);
	for (int i = 0; i < numTriangles; i++) {
		std::vector<cv::Point2f> triangle;
		for (int j = 0; j < 3; j++) {
			triangle.push_back(points[triangleVertices[i][j]]);
		}
		trianglerPts[i] = triangle;
	}
}

void paint_triangles(cv::Mat &img, const std::vector<std::vector<cv::Point2f>> &triangles) {
	int numTriangles = triangles.size();

	for (int i = 0; i < numTriangles; i++) {
		std::vector<cv::Point> poly(3);

		for (int j = 0; j < 3; j++) {
			poly[j] = cv::Point(cvRound(triangles[i][j].x), cvRound(triangles[i][j].y));
		}
		cv::fillConvexPoly(img, poly, cv::Scalar(i + 1));
	}
}

void solve_homography(const std::vector<cv::Point2f> &srcPts1, const std::vector<cv::Point2f> &srcPts2, cv::Mat &homography) {
	assert(srcPts1.size() == srcPts2.size());
	homography = points_to_homogenous_mat(srcPts2) * points_to_homogenous_mat(srcPts1).inv();
}

void solve_homography(const std::vector<std::vector<cv::Point2f>> &srcPts1,
		const std::vector<std::vector<cv::Point2f>> &srcPts2,
		std::vector<cv::Mat> &hmats) {
	assert(srcPts1.size() == srcPts2.size());

	int ptsNum = srcPts1.size();
	hmats.clear();
	hmats.reserve(ptsNum);
	for (int i = 0; i < ptsNum; i++) {
		cv::Mat homography;
		solve_homography(srcPts1[i], srcPts2[i], homography);
		hmats.push_back(homography);
	}
}

void morph_homography(const cv::Mat &Hom, cv::Mat &MorphHom1, cv::Mat &MorphHom2, float blend_ratio) {
	cv::Mat invHom = Hom.inv();
	MorphHom1 = cv::Mat::eye(3, 3, CV_32FC1) * (1.0 - blend_ratio) + Hom * blend_ratio;
	MorphHom2 = cv::Mat::eye(3, 3, CV_32FC1) * blend_ratio + invHom * (1.0 - blend_ratio);
}

void morph_homography(const std::vector<cv::Mat> &homs,
		std::vector<cv::Mat> &morphHoms1,
		std::vector<cv::Mat> &morphHoms2,
		float blend_ratio) {
	int numHoms = homs.size();
	morphHoms1.resize(numHoms);
	morphHoms2.resize(numHoms);
	for (int i = 0; i < numHoms; i++) {
		morph_homography(homs[i], morphHoms1[i], morphHoms2[i], blend_ratio);
	}
}

void create_map(const cv::Mat &triangleMap, const std::vector<cv::Mat> &homMatrices, cv::Mat &mapx, cv::Mat &mapy) {
	assert(triangleMap.type() == CV_32SC1);

	// Allocate cv::Mat for the map
	mapx.create(triangleMap.size(), CV_32FC1);
	mapy.create(triangleMap.size(), CV_32FC1);

	// Compute inverse matrices
	std::vector<cv::Mat> invHomMatrices(homMatrices.size());
	for (size_t i = 0; i < homMatrices.size(); i++) {
		invHomMatrices[i] = homMatrices[i].inv();
	}

	for (int y = 0; y < triangleMap.rows; y++) {
		for (int x = 0; x < triangleMap.cols; x++) {
			int idx = triangleMap.at<int>(y, x) - 1;
			if (idx >= 0) {
				cv::Mat H = invHomMatrices[triangleMap.at<int>(y, x) - 1];
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

Point2f calculate_line_point(double x1, double y1, double x2, double y2, double distance) {
	double vx = x2 - x1; // x vector
	double vy = y2 - y1; // y vector
	double mag = hypot(vx, vy); // length

	vx /= mag;
	vy /= mag;

	double px = x1 + vx * distance;
	double py = y1 + vy * distance;
	return {(float)px, (float)py};
}

void find_contours(const Mat &img1, const Mat &img2, std::vector<Mat> &dst1, std::vector<Mat> &dst2) {
	std::vector<std::vector<cv::Point>> contours1;
	std::vector<std::vector<cv::Point>> contours2;
	Mat grey1, grey2;
	Mat thresh1, thresh2;
	vector<Vec4i> hierarchy1;
	vector<Vec4i> hierarchy2;

	cvtColor(img1, grey1, cv::COLOR_RGB2GRAY);
	cvtColor(img2, grey2, cv::COLOR_RGB2GRAY);

	std::vector<std::vector<std::vector<cv::Point>>> collected1;
	std::vector<std::vector<std::vector<cv::Point>>> collected2;

	for (off_t i = 0; i < 9; ++i) {
		cv::threshold(grey1, thresh1, std::min(255, (int) round((i + 1) * 25 * contour_sensitivity)), 255, 0);
		cv::findContours(thresh1, contours1, hierarchy1, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);
		collected1.push_back(contours1);
	}

	for (off_t i = 0; i < 9; ++i) {
		cv::threshold(grey2, thresh2, std::min(255, (int) round((i + 1) * 25 * contour_sensitivity)), 255, 0);
		cv::findContours(thresh2, contours2, hierarchy2, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);
		collected2.push_back(contours2);
	}

	dst1.clear();
	dst2.clear();
	dst1.resize(collected1.size());
	dst2.resize(collected1.size());

	for (size_t i = 0; i < collected1.size(); ++i) {
		Mat &cont1 = dst1[i];
		Mat &cont2 = dst2[i];
		cvtColor(thresh1, cont1, cv::COLOR_GRAY2RGB);
		cvtColor(thresh2, cont2, cv::COLOR_GRAY2RGB);

		for (size_t i = 0; i < contours1.size(); ++i)
			cv::drawContours(cont1, contours1, i, { 255, 0, 0 }, 1, cv::LINE_8, hierarchy1, 0);

		for (size_t i = 0; i < contours2.size(); ++i)
			cv::drawContours(cont2, contours2, i, { 255, 0, 0 }, 1, cv::LINE_8, hierarchy2, 0);

		cvtColor(cont1, cont1, cv::COLOR_RGB2GRAY);
		cvtColor(cont2, cont2, cv::COLOR_RGB2GRAY);
	}
}

void find_matches(Mat &orig1, Mat &orig2, std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2) {
	std::vector<Mat> contours1, contours2;
	find_contours(orig1, orig2, contours1, contours2);

	for (size_t i = 0; i < contours1.size(); ++i) {
		auto matches = find_matches(contours1[i], contours2[i]);
		srcPoints1.insert(srcPoints1.end(), matches.first.begin(), matches.first.end());
		srcPoints2.insert(srcPoints2.end(), matches.second.begin(), matches.second.end());
	}
	std::cerr << "contour points: " << srcPoints1.size() << std::endl;

	Mat matMatches;
	Mat grey1, grey2;
	cvtColor(orig1, grey1, cv::COLOR_RGB2GRAY);
	cvtColor(orig2, grey2, cv::COLOR_RGB2GRAY);
	draw_matches(grey1, grey2, matMatches, srcPoints1, srcPoints2);
	imshow("matches", matMatches);

}
void pair_points_by_proximity(std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2, int cols, int rows) {
	std::set<Point2f, LessPointOp> setpt2;
	for (auto pt2 : srcPoints2) {
		setpt2.insert(pt2);
	}

	std::vector<cv::Point2f> tmp1;
	std::vector<cv::Point2f> tmp2;
	double maxLen = hypot(cols, rows) / max_pair_len_divider;
	for (size_t i = 0; i < srcPoints1.size(); ++i) {
		auto pt1 = srcPoints1[i];
		double dist = 0;
		double currentMinDist = std::numeric_limits<double>::max();

		Point2f closest(-1, -1);
		for (auto pt2 : setpt2) {
			dist = hypot(pt2.x - pt1.x, pt2.y - pt1.y);

			if (dist < currentMinDist) {
				currentMinDist = dist;
				closest = pt2;
			}
		}

		if (closest.x == -1 && closest.y == -1)
			continue;

		dist = hypot(closest.x - pt1.x, closest.y - pt1.y);
		if (dist < maxLen) {
			tmp1.push_back(pt1);
			tmp2.push_back(closest);
			setpt2.erase(closest);
		}
	}

	  std::vector<Segment_2> segs;
	  for(size_t i = 0; i < tmp1.size(); i++) {
		  if(tmp1[i] != tmp2[i])
			  segs.push_back(Segment_2(Point_2(tmp1[i].x, tmp1[i].y), Point_2(tmp2[i].x, tmp2[i].y)));
	  }
	  std::list<Segment_2> sub_segs;

	  CGAL::compute_subcurves(segs.data(), segs.data() + segs.size(), std::back_inserter(sub_segs));

	  tmp1.clear();
	  tmp2.clear();
	  for(const Segment_2& subseg : sub_segs) {
	    double x1 = CGAL::to_double(subseg.source()[0]);
	    double y1 = CGAL::to_double(subseg.source()[1]);
	    double x2 = CGAL::to_double(subseg.target()[0]);
	    double y2 = CGAL::to_double(subseg.target()[1]);
	    tmp1.push_back(Point2f(x1,y1));
	    tmp2.push_back(Point2f(x2,y2));
	  }

	assert(tmp1.size() == tmp2.size());

	srcPoints1 = tmp1;
	srcPoints2 = tmp2;

	for (auto pt : srcPoints1) {
		assert(!isinf(pt.x) && !isinf(pt.y));
		assert(!isnan(pt.x) && !isnan(pt.y));
		assert(pt.x >= 0 && pt.y >= 0);
		assert(pt.x < cols && pt.y < rows);
	}

	for (auto pt : srcPoints2) {
		assert(!isinf(pt.x) && !isinf(pt.y));
		assert(!isnan(pt.x) && !isnan(pt.y));
		assert(pt.x >= 0 && pt.y >= 0);
		assert(pt.x < cols && pt.y < rows);
	}
}

void chop_long_travel_paths(std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2, int cols, int rows) {
	std::set<Point2f, LessPointOp> setpt2;
	for (auto pt2 : srcPoints2) {
		setpt2.insert(pt2);
	}

	std::vector<cv::Point2f> tmp1;
	std::vector<cv::Point2f> tmp2;
	double maxLen = max_chop_len;
	for (size_t i = 0; i < srcPoints1.size(); ++i) {
		auto pt1 = srcPoints1[i];
		double dist = 0;
		double currentMinDist = std::numeric_limits<double>::max();

		Point2f closest(-1, -1);
		bool erased = false;
		for (auto pt2 : setpt2) {
			dist = hypot(pt2.x - pt1.x, pt2.y - pt1.y);
			if (dist < currentMinDist) {
				currentMinDist = dist;
				closest = pt2;
			}
		}

		if (erased)
			continue;

		if (closest.x == -1 && closest.y == -1)
			continue;

		dist = hypot(closest.x - pt1.x, closest.y - pt1.y);
		if (dist < maxLen) {
			tmp1.push_back(pt1);
			tmp2.push_back(closest);
			setpt2.erase(closest);
		} else {
			Point2f newPt = calculate_line_point(pt1.x, pt1.y, closest.x, closest.y, maxLen - 1);
			setpt2.insert(newPt);
			Point2f oldPt;
			while (hypot(closest.x - newPt.x, closest.y - newPt.y) >= maxLen) {
				oldPt = newPt;
				newPt = calculate_line_point(newPt.x, newPt.y, closest.x, closest.y, maxLen - 1);
				tmp1.push_back(pt1);
				tmp2.push_back(newPt);
			}
			--i;
		}
	}

	assert(tmp1.size() == tmp2.size());

	srcPoints1 = tmp1;
	srcPoints2 = tmp2;

	for (auto pt : srcPoints1) {
		assert(!isinf(pt.x) && !isinf(pt.y));
		assert(!isnan(pt.x) && !isnan(pt.y));
		assert(pt.x >= 0 && pt.y >= 0);
		assert(pt.x < cols && pt.y < rows);
	}

	for (auto pt : srcPoints2) {
		assert(!isinf(pt.x) && !isinf(pt.y));
		assert(!isnan(pt.x) && !isnan(pt.y));
		assert(pt.x >= 0 && pt.y >= 0);
		assert(pt.x < cols && pt.y < rows);
	}
}

void add_corners(std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2, MatSize sz) {
	float w = sz().width - 1;
	float h = sz().height - 1;
	srcPoints1.push_back(cv::Point2f(0, 0));
	srcPoints1.push_back(cv::Point2f(w, 0));
	srcPoints1.push_back(cv::Point2f(0, h));
	srcPoints1.push_back(cv::Point2f(w, h));
	srcPoints2.push_back(cv::Point2f(0, 0));
	srcPoints2.push_back(cv::Point2f(w, 0));
	srcPoints2.push_back(cv::Point2f(0, h));
	srcPoints2.push_back(cv::Point2f(w, h));
}

void draw_optical_flow(const Mat &img1, const Mat &img2, Mat &dst) {
	UMat flowUmat;
	Mat flow;
	Mat grey1, grey2;
	cvtColor(img1, grey1, cv::COLOR_RGB2GRAY);
	cvtColor(img2, grey2, cv::COLOR_RGB2GRAY);
	calcOpticalFlowFarneback(grey1, grey2, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
	flowUmat.copyTo(flow);
	dst = img1.clone();
	// By y += 5, x += 5 you can specify the grid
	for (int y = 0; y < dst.rows; y += 20) {
		for (int x = 0; x < dst.cols; x += 20) {
			// get the flow from y, x position * 10 for better visibility
			const Point2f flowatxy = flow.at<Point2f>(y, x) * 10;
			// draw line at flow direction
			line(dst, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255, 0, 0));
			// draw initial point
			circle(dst, Point(x, y), 1, Scalar(0, 0, 255), -1);
		}
	}
}

void prepare_matches(Mat &origImg1, Mat &origImg2, const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2) {
	//edit matches
	std::cerr << "prepare: " << srcPoints1.size() << " -> ";
	pair_points_by_proximity(srcPoints1, srcPoints2, img1.cols, img1.rows);
	std::cerr << "pair: " << srcPoints1.size() << " -> ";
	chop_long_travel_paths(srcPoints1, srcPoints2, img1.cols, img1.rows);
	std::cerr << "chop: " << srcPoints1.size() << " -> ";

	std::vector<std::tuple<KeyPoint, KeyPoint, double>> edges;
	edges.reserve(1000);
	Point2f p1, p2;
	for (size_t i = 0; i < srcPoints1.size(); ++i) {
		auto &pt1 = srcPoints1[i];
		auto &pt2 = srcPoints2[i];
		edges.push_back( { { pt1, 1 }, { pt2, 1 }, distance(pt1, Point2f(pt2.x + img1.cols, pt2.y)) });
	}
	std::vector<KeyPoint> kpv1;
	std::vector<KeyPoint> kpv2;
	length_test(edges, kpv1, kpv2, img1.cols);

	srcPoints1.clear();
	for (auto kp : kpv1) {
		srcPoints1.push_back(kp.pt);
	}
	srcPoints2.clear();
	for (auto kp : kpv2) {
		srcPoints2.push_back(kp.pt);
	}
	std::cerr << "length test: " << srcPoints1.size() << " -> ";

	angle_test(srcPoints1, srcPoints2, img1.cols);
	std::cerr << "angle test: " << srcPoints1.size() << std::endl;

	Mat matMatches;
	Mat grey1, grey2;
	cvtColor(origImg1, grey1, cv::COLOR_RGB2GRAY);
	cvtColor(origImg2, grey2, cv::COLOR_RGB2GRAY);
	draw_matches(grey1, grey2, matMatches, srcPoints1, srcPoints2);
	imshow("matches reduced", matMatches);

	if (srcPoints1.size() > srcPoints2.size())
		srcPoints1.resize(srcPoints2.size());
	else
		srcPoints2.resize(srcPoints1.size());

	add_corners(srcPoints1, srcPoints2, origImg1.size);
}

double morph_images(Mat &origImg1, Mat &origImg2, cv::Mat &dst, const cv::Mat &last, std::vector<cv::Point2f> srcPoints1, std::vector<cv::Point2f> srcPoints2, float shapeRatio = 0.5, float colorRatio = -1) {
	//morph based on matches
	cv::Size SourceImgSize(origImg1.cols, origImg1.rows);
	cv::Subdiv2D subDiv1(cv::Rect(0, 0, SourceImgSize.width, SourceImgSize.height));
	cv::Subdiv2D subDiv2(cv::Rect(0, 0, SourceImgSize.width, SourceImgSize.height));
	cv::Subdiv2D subDivMorph(cv::Rect(0, 0, SourceImgSize.width, SourceImgSize.height));
	for (auto pt : srcPoints1) {
		assert(!isinf(pt.x) && !isinf(pt.y));
		assert(!isnan(pt.x) && !isnan(pt.y));
		assert(pt.x >= 0 && pt.y >= 0);
		assert(pt.x < origImg1.cols && pt.y < origImg1.rows);
//		std::cerr << pt << std::endl;
		subDiv1.insert(pt);
	}
	for (auto pt : srcPoints2) {
		assert(!isinf(pt.x) && !isinf(pt.y));
		assert(!isnan(pt.x) && !isnan(pt.y));
		assert(pt.x >= 0 && pt.y >= 0);
		assert(pt.x < origImg1.cols && pt.y < origImg1.rows);
//		std::cerr << pt << std::endl;
		subDiv2.insert(pt);
	}

	std::vector<cv::Point2f> morphedPoints;
	morph_points(srcPoints1, srcPoints2, morphedPoints, shapeRatio);
	assert(srcPoints1.size() == srcPoints2.size() && srcPoints2.size() == morphedPoints.size());
	for (auto pt : morphedPoints) {
		assert(!isinf(pt.x) && !isinf(pt.y));
		assert(!isnan(pt.x) && !isnan(pt.y));
		assert(pt.x >= 0 && pt.y >= 0);
		assert(pt.x < origImg1.cols && pt.y < origImg1.rows);
//		std::cerr << pt << std::endl;
		subDivMorph.insert(pt);
	}

	// Get the ID list of corners of Delaunay traiangles.
	std::vector<cv::Vec3i> triangleIndices;
	get_triangle_indices(subDivMorph, morphedPoints, triangleIndices);

	// Get coordinates of Delaunay corners from ID list
	std::vector<std::vector<cv::Point2f>> triangleSrc1, triangleSrc2, triangleMorph;
	make_triangler_points(triangleIndices, srcPoints1, triangleSrc1);
	make_triangler_points(triangleIndices, srcPoints2, triangleSrc2);
	make_triangler_points(triangleIndices, morphedPoints, triangleMorph);

	// Create a map of triangle ID in the morphed image.
	cv::Mat triMap = cv::Mat::zeros(SourceImgSize, CV_32SC1);
	paint_triangles(triMap, triangleMorph);

	// Compute Homography matrix of each triangle.
	std::vector<cv::Mat> homographyMats, morphHom1, morphHom2;
	solve_homography(triangleSrc1, triangleSrc2, homographyMats);
	morph_homography(homographyMats, morphHom1, morphHom2, shapeRatio);

	cv::Mat trImg1;
	cv::Mat trans_map_x1, trans_map_y1;
	create_map(triMap, morphHom1, trans_map_x1, trans_map_y1);
	cv::remap(origImg1, trImg1, trans_map_x1, trans_map_y1, cv::INTER_LINEAR);

	cv::Mat trImg2;
	cv::Mat trMapX2, trMapY2;
	create_map(triMap, morphHom2, trMapX2, trMapY2);
	cv::remap(origImg2, trImg2, trMapX2, trMapY2, cv::INTER_LINEAR);

	// Blend 2 input images
	float blend = (colorRatio < 0) ? shapeRatio : colorRatio;
	dst = trImg1 * (1.0 - blend) + trImg2 * blend;
	Mat analysis = dst.clone();
	Mat prev = last.clone();
	if (prev.empty())
		prev = dst.clone();
	draw_morph_analysis(dst, prev, analysis, SourceImgSize, subDiv1, subDiv2, subDivMorph, { 0, 0, 255 });
	imshow("analysis", analysis);
	return 0;
}

double ease_in_out_sine(double x) {
	return -(cos(M_PI * x) - 1) / 2;
}

Point2f rotate_point(const Point2f &center, const Point2f trabant, float angle) {
	float s = sin(angle);
	float c = cos(angle);
	Point2f newPoint = trabant;
	// translate point back to origin:
	newPoint.x -= center.x;
	newPoint.y -= center.y;

	// rotate point
	float xnew = newPoint.x * c - newPoint.y * s;
	float ynew = newPoint.x * s + newPoint.y * c;

	// translate point back:
	newPoint.x = xnew + center.x;
	newPoint.y = ynew + center.y;
	return newPoint;
}

int main(int argc, char **argv) {
	using std::string;
	srand(time(NULL));
	double numberOfFrames = number_of_frames;
	double maxLenDeviation = max_len_deviation;
	double maxAngDeviation = max_ang_deviation;
	double maxPairLenDivider = max_pair_len_divider;
	double maxChopLen = max_chop_len;
	double contSensitivity = contour_sensitivity;
	std::vector<string> imageFiles;
	string outputFile = "output.mkv";

	po::options_description genericDesc("Options");
	genericDesc.add_options()
	("frames,f", po::value<double>(&numberOfFrames)->default_value(numberOfFrames), "The number of frames to generate")
	("lendev,l", po::value<double>(&maxLenDeviation)->default_value(maxLenDeviation), "The maximum length deviation in percent for the length test")
	("angdev,a", po::value<double>(&maxAngDeviation)->default_value(maxAngDeviation), "The maximum angular deviation in percent for the angle test")
	("pairlen,p", po::value<double>(&maxPairLenDivider)->default_value(maxPairLenDivider), "The divider that controls the maximum distance (diagonal/divider) for point pairs")
	("choplen,c", po::value<double>(&maxChopLen)->default_value(maxChopLen), "The interval in which traversal paths (point pairs) are chopped")
	("sensitivity,s", po::value<double>(&contSensitivity)->default_value(contSensitivity), "How sensitive to contours the matcher showed be (values less than 1.0 make it more sensitive)")
	("outfile,o", po::value<string>(&outputFile)->default_value(outputFile), "The name of the video file to write to")
	("help,h", "Print help message");

	po::options_description hidden("Hidden options");
	hidden.add_options()("files", po::value<std::vector<string>>(&imageFiles), "image files");

	po::options_description cmdline_options;
	cmdline_options.add(genericDesc).add(hidden);

	po::positional_options_description p;
	p.add("files", -1);

	po::options_description visible;
	visible.add(genericDesc);

	po::variables_map vm;
	po::store(
			po::command_line_parser(argc, argv).options(cmdline_options).positional(
					p).run(), vm);
	po::notify(vm);

	if (vm.count("help") || imageFiles.empty()) {
		std::cerr << "Usage: poppy [options] <imageFiles>..." << std::endl;
		std::cerr << visible;
		return 0;
	}

	for (auto p : imageFiles) {
		if (!std::filesystem::exists(p))
			throw std::runtime_error("File doesn't exist: " + p);
	}
	number_of_frames = numberOfFrames;
	max_len_deviation = maxLenDeviation;
	max_ang_deviation = maxAngDeviation;
	max_pair_len_divider = maxPairLenDivider;
	max_chop_len = maxChopLen;
	contour_sensitivity = contSensitivity;
	Mat image1;
	try {
		image1 = imread(imageFiles[0], cv::IMREAD_COLOR);
		if (image1.empty()) {
			std::cerr << "Can't read (invalid?) image file: " + imageFiles[0] << std::endl;
			exit(2);
		}
	} catch (...) {
		std::cerr << "Can't read (invalid?) image file: " + imageFiles[0] << std::endl;
		exit(2);
	}

	Mat image2;
	Mat orig1;
	Mat orig2;
	VideoWriter output(outputFile, VideoWriter::fourcc('H', '2', '6', '4'), 30,
			Size(image1.cols, image1.rows));

	for (size_t i = 1; i < imageFiles.size(); ++i) {
		Mat image2;
		try {
			image2 = imread(imageFiles[i], cv::IMREAD_COLOR);
			if (image2.empty()) {
				std::cerr << "Can't read (invalid?) image file: " + imageFiles[i] << std::endl;
				exit(2);
			}
		} catch (...) {
			std::cerr << "Can't read (invalid?) image file: " + imageFiles[i] << std::endl;
			exit(2);
		}

		orig1 = image1.clone();
		orig2 = image2.clone();

		Mat morphed;

		std::vector<Point2f> srcPoints1;
		std::vector<Point2f> srcPoints2;
		std::cerr << "matching: " << imageFiles[i - 1] << " -> " << imageFiles[i] << " ..." << std::endl;
		find_matches(orig1, orig2, srcPoints1, srcPoints2);
		prepare_matches(orig1, orig2, image1, image2, srcPoints1, srcPoints2);

		float step = 1.0 / number_of_frames;
		for (size_t j = 0; j < number_of_frames; ++j) {
			std::cerr << int((j / number_of_frames) * 100.0) << "%\r";

			morph_images(orig1, orig2, morphed, morphed.clone(), srcPoints1, srcPoints2, ease_in_out_sine((j + 1) * step), (j + 1) * step);
			image1 = morphed.clone();
			output.write(morphed);

			imshow("morphed", morphed);
			waitKey(1);
		}
		morphed.release();
		srcPoints1.clear();
		srcPoints2.clear();

		image1 = image2.clone();
	}
	return 0;
}
