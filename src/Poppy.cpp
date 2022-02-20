#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <list>
#include <unordered_set>
#include <filesystem>

#include <boost/program_options.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

double number_of_frames = 60;
double max_len_deviation = 20;
double max_ang_deviation = 0.3;
double max_pair_len_divider = 20;
double max_chop_len = 2;
int contour_sensitivity = 40;

using namespace cv;
using std::vector;
using std::chrono::microseconds;

namespace po = boost::program_options;

typedef unsigned char sample_t;

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

void angle_test(std::vector<KeyPoint>& kpv1, std::vector<KeyPoint>& kpv2, int cols) {
	double maxDeviationPercent = max_ang_deviation;
	double avg = 0;
	double total = 0;

	for(size_t i = 0; i < kpv1.size(); ++i) {
		total += M_PI + std::atan2(kpv2[i].pt.y - kpv1[i].pt.y, (cols + kpv2[i].pt.x) - kpv1[i].pt.x);
	}

	avg = total / kpv1.size();
	double dev = avg / (100 / maxDeviationPercent);
	double min = avg - (dev / 2);
	double max = min + dev;

	std::vector<KeyPoint> new1;
	std::vector<KeyPoint> new2;

	for(size_t i = 0; i < kpv1.size(); ++i) {
		double angle = M_PI + std::atan2(kpv2[i].pt.y - kpv1[i].pt.y, (cols + kpv2[i].pt.x) - kpv1[i].pt.x);

		if(angle > min && angle < max) {
			new1.push_back(kpv1[i]);
			new2.push_back(kpv2[i]);
		}
	}
	kpv1 = new1;
	kpv2 = new2;
	std::cerr << "angle matches: " << new1.size() << " -> ";
}

void length_test(std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols) {
	double maxDeviationPercent = max_len_deviation;

	std::vector<std::tuple<KeyPoint, KeyPoint, double>> edges;
	edges.reserve(1000);
	Point2f p1, p2;
	for (auto &kp1 : kpv1) {
		for (auto &kp2 : kpv2) {
			edges.push_back({ kp1, kp2, distance(kp1.pt, Point2f(kp2.pt.x + cols, kp2.pt.y)) });
		}
	}

	double avg = 0;
	double total = 0;

	for(auto e : edges) {
		total += std::get<2>(e);
	}

	avg = total / edges.size();
	double dev = avg / (100 / maxDeviationPercent);
	double min = avg - (dev / 2);
	double max = min + dev;


	kpv1.clear();
	kpv2.clear();

	for(auto e : edges) {
		double len = std::get<2>(e);
		if(len > min && len < max) {
			kpv1.push_back(std::get<0>(e));
			kpv2.push_back(std::get<1>(e));
		}
	}

	std::cerr << "length matches: " << kpv1.size() << " -> ";
}

// Draw delaunay triangles
static void draw_delaunay(Mat &dst, const Size &size, Subdiv2D &subdiv, Scalar delaunay_color) {
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);
	Rect rect(0, 0, size.width, size.height);

	for (size_t i = 0; i < triangleList.size(); i++) {
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

		// Draw rectangles completely inside the image.
		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
				{
			line(dst, pt[0], pt[1], delaunay_color, 1, cv::LINE_AA, 0);
			line(dst, pt[1], pt[2], delaunay_color, 1, cv::LINE_AA, 0);
			line(dst, pt[2], pt[0], delaunay_color, 1, cv::LINE_AA, 0);
		}
	}
}

void draw_matches(const Mat& src1, const Mat& src2, Mat& dst, std::vector<KeyPoint>& kpv1, std::vector<KeyPoint>& kpv2) {
	Mat grey1 = src1, grey2 = src2;

	Mat images = cv::Mat::zeros({grey1.cols * 2, grey1.rows}, CV_8UC1);
	Mat lines = cv::Mat::ones({grey1.cols * 2, grey1.rows}, CV_8UC1);

	grey1.copyTo(images(cv::Rect( 0, 0, grey1.cols, grey1.rows)));
	grey2.copyTo(images(cv::Rect( grey1.cols, 0, grey1.cols, grey1.rows)));

	for(size_t i = 0; i < kpv1.size(); ++i) {
		Point2f pt2 = kpv2[i].pt;
		pt2.x += src1.cols;
		line(lines, kpv1[i].pt, pt2, {127}, 1, cv::LINE_AA, 0);
	}

	images += 1;
	Mat result = images.mul(lines);
	cvtColor(result, dst, COLOR_GRAY2RGB);
}

std::pair<std::vector<Point2f>, std::vector<Point2f>> find_matches(const Mat &grey1, const Mat &grey2) {
	Mat edge1;
	Mat edge2;

	cv::Ptr<cv::ORB> detector = cv::ORB::create(1000);
	cv::Ptr<cv::ORB> extractor = cv::ORB::create();

	std::vector<KeyPoint> keypoints1, keypoints2;

	Mat descriptors1, descriptors2;
	detector->detect(grey1, keypoints1);
	detector->detect(grey2, keypoints2);

	detector->compute(grey1, keypoints1, descriptors1);
	detector->compute(grey2, keypoints2, descriptors2);

	Mat matMatches;

	length_test(keypoints1, keypoints2, grey1.cols);
	angle_test(keypoints1, keypoints2, grey1.cols);

//	draw_matches(grey1, grey2, matMatches, keypoints1, keypoints2);
//	imshow("len", matMatches);


	std::vector<Point2f> points1;
	std::vector<Point2f> points2;
	for(auto pt1 : keypoints1) {
		points1.push_back(pt1.pt);
	}

	for(auto pt2 : keypoints2) {
		points2.push_back(pt2.pt);
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

void morph_homography(const cv::Mat& Hom, cv::Mat& MorphHom1, cv::Mat& MorphHom2, float blend_ratio)
		{
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

double l2_error(const Mat &a, const Mat &b) {
	if (a.rows > 0 && a.rows == b.rows && a.cols > 0 && a.cols == b.cols) {
		double errorL2 = cv::norm(a, b, cv::NORM_L2);
		double similarity = errorL2 / (double) (a.rows * a.cols);
		return similarity;
	}
	else {
		return std::numeric_limits<double>::max();
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

void find_contours(const Mat &img1, const Mat &img2, std::vector<Point2f> &dstPoints1, std::vector<Point2f> &dstPoints2, Mat &dst1, Mat &dst2) {
	std::vector<std::vector<cv::Point>> contours1;
	std::vector<std::vector<cv::Point>> contours2;
	Mat grey1, grey2;
	Mat thresh1, thresh2;
	vector<Vec4i> hierarchy1;
	vector<Vec4i> hierarchy2;

	//FIXME
	static double t = contour_sensitivity;
	static bool first = true;
	double tmp;
	cvtColor(img1, grey1, cv::COLOR_RGB2GRAY);
	cvtColor(img2, grey2, cv::COLOR_RGB2GRAY);

	canny_threshold(grey1, thresh1, t);
	canny_threshold(grey2, thresh2, t);

	dst1 = thresh1.clone();
	dst2 = thresh2.clone();

	cv::findContours(thresh1, contours1, hierarchy1, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);
	cv::findContours(thresh2, contours2, hierarchy2, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);
	if (first) {
		tmp = t;
		while (contours2.size() * 1.2 < contours1.size() && tmp > 0) {
			assert(!contours1.empty() && !contours1.empty());
			canny_threshold(grey2, thresh2, --tmp);
			cv::findContours(thresh2, contours2, hierarchy2, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);
		}

		tmp = t;
		while (contours1.size() * 1.2 < contours2.size() && tmp < 255) {
			assert(!contours1.empty() && !contours1.empty());
			canny_threshold(grey2, thresh2, ++tmp);
			cv::findContours(thresh2, contours2, hierarchy2, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);
		}
		t = tmp;
	}
	first = false;
	std::cerr << "contours: " << contours1.size() << " -> ";

	Mat cont1;
	Mat cont2;
	cvtColor(thresh1, cont1, cv::COLOR_GRAY2RGB);
	cvtColor(thresh2, cont2, cv::COLOR_GRAY2RGB);

	for (size_t i = 0; i < contours1.size(); ++i)
		cv::drawContours(cont1, contours1, i, { 255, 0, 0 }, 2, cv::LINE_8, hierarchy1, 0);

	for (size_t i = 0; i < contours2.size(); ++i)
		cv::drawContours(cont2, contours2, i, { 255, 0, 0 }, 2, cv::LINE_8, hierarchy2, 0);

	imshow("cont1", cont1);
	imshow("cont2", cont2);
}

void pair_points_by_proximity(std::vector<cv::Point2f>& srcPoints1, std::vector<cv::Point2f>& srcPoints2, int cols, int rows) {
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

void chop_long_travel_paths(std::vector<cv::Point2f>& srcPoints1, std::vector<cv::Point2f>& srcPoints2, int cols, int rows) {
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
			assert(false);

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

void add_corners(std::vector<cv::Point2f>& srcPoints1, std::vector<cv::Point2f>& srcPoints2, MatSize sz) {
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
double morph_images(Mat& origImg1, Mat& origImg2, const cv::Mat &img1, const cv::Mat &img2, cv::Mat &dst, std::vector<Point2f> &prevPoints, std::vector<cv::Point2f> &dstPoints, float shapeRatio = 0.5, float colorRatio = -1) {
	std::vector<cv::Point2f> srcPoints1;
	std::vector<cv::Point2f> srcPoints2;

	//find matches
	Mat contours1, contours2;
	find_contours(origImg1, origImg2, srcPoints1, srcPoints2, contours1, contours2);
	auto matches = find_matches(contours1, contours2);
	srcPoints1 = matches.first;
	srcPoints2 = matches.second;

	//edit matches
	pair_points_by_proximity(srcPoints1, srcPoints2, img1.cols, img1.rows);
	chop_long_travel_paths(srcPoints1, srcPoints2, img1.cols, img1.rows);

	if (srcPoints1.size() > srcPoints2.size())
		srcPoints1.resize(srcPoints2.size());
	else
		srcPoints2.resize(srcPoints1.size());

	add_corners(srcPoints1, srcPoints2, origImg1.size);

	//morph based on matches
	int numPoints = srcPoints1.size();
	cv::Size SourceImgSize(srcPoints1[numPoints - 1].x + 1, srcPoints1[numPoints - 1].y + 1);
	cv::Subdiv2D subDiv1(cv::Rect(0, 0, SourceImgSize.width, SourceImgSize.height));
	cv::Subdiv2D subDiv2(cv::Rect(0, 0, SourceImgSize.width, SourceImgSize.height));
	cv::Subdiv2D subDivMorph(cv::Rect(0, 0, SourceImgSize.width, SourceImgSize.height));
	subDiv1.insert(srcPoints1);
	subDiv2.insert(srcPoints2);

	std::vector<cv::Point2f> morphedPoints;
	morph_points(srcPoints1, srcPoints2, morphedPoints, shapeRatio);

	for (auto pt : morphedPoints) {
		assert(!isinf(pt.x) && !isinf(pt.y));
		assert(!isnan(pt.x) && !isnan(pt.y));
		assert(pt.x >= 0 && pt.y >= 0);
		assert(pt.x < img1.cols && pt.y < img1.rows);
	}
	std::cerr << "morph points: " << morphedPoints.size() << std::endl << std::flush;

	subDivMorph.insert(morphedPoints);

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
//	double error = l2_error(img2, dst);
//	std::cerr << "Error: " << error << std::endl;
	Mat delaunay = dst.clone();
	draw_delaunay(delaunay, SourceImgSize, subDiv1, { 255, 0, 0 });
	draw_delaunay(delaunay, SourceImgSize, subDiv2, { 0, 255, 0 });
	draw_delaunay(delaunay, SourceImgSize, subDivMorph, { 0, 0, 255 });
	imshow("delaunay", delaunay);
	dstPoints.clear();
	dstPoints.insert(dstPoints.end(), morphedPoints.begin(), morphedPoints.end() - 4);

	return 0;
}

int main(int argc, char **argv) {
	using std::string;
	srand(time(NULL));
	double numberOfFrames = number_of_frames;
	double maxLenDeviation = max_len_deviation;
	double maxAngDeviation = max_ang_deviation;
	double maxPairLenDivider = max_pair_len_divider;
	double maxChopLen = max_chop_len;
	int contSensitivity = contour_sensitivity;
	std::vector<string> imageFiles;
	string outputFile = "output.mkv";

	po::options_description genericDesc("Options");
	genericDesc.add_options()
			("frames,f", po::value<double>(&numberOfFrames)->default_value(numberOfFrames), "The number of frames to generate")
			("lendev,l", po::value<double>(&maxLenDeviation)->default_value(maxLenDeviation), "The maximum length deviation in percent for the length test")
			("angdev,a", po::value<double>(&maxAngDeviation)->default_value(maxAngDeviation), "The maximum angular deviation in percent for the angle test")
			("pairlen,p", po::value<double>(&maxPairLenDivider)->default_value(maxPairLenDivider), "The divider that controls the maximum distance (diagonal/divider) for point pairs")
			("choplen,c", po::value<double>(&maxChopLen)->default_value(maxChopLen), "The interval in which traversal paths (point pairs) are chopped")
			("sensitivity,s", po::value<int>(&contSensitivity)->default_value(contSensitivity), "How sensitive to contours the matcher showed be (1-255)")
			("outfile,o", po::value<string>(&outputFile)->default_value(outputFile), "The name of the video file to write to")
			("help,h","Print help message");

	po::options_description hidden("Hidden options");
	hidden.add_options()("files",po::value<std::vector<string>>(&imageFiles), "image files");

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

	for(auto p : imageFiles) {
		if(!std::filesystem::exists(p))
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
		if(image1.empty()) {
			std::cerr << "Can't read (invalid?) image file: " + imageFiles[0] << std::endl;
			exit(2);
		}
	} catch(...) {
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
			if(image2.empty()) {
				std::cerr << "Can't read (invalid?) image file: " + imageFiles[i] << std::endl;
				exit(2);
			}
		} catch(...) {
			std::cerr << "Can't read (invalid?) image file: " + imageFiles[i] << std::endl;
			exit(2);
		}
		orig1 = image1.clone();
		orig2 = image2.clone();

		Mat morphed;

		std::vector<Point2f> inputPts;
		std::vector<Point2f> outputPts;

		float step = 1.0 / number_of_frames;
		for (size_t j = 0; j < 10; ++j) {
			output.write(orig1);
		}
		for (size_t j = 0; j < number_of_frames; ++j) {
			std::cerr  << double((j / number_of_frames) * 100.0) << '%' << std::endl;
			morph_images(orig1, orig2, image1, image2, morphed, inputPts, outputPts, (j + 1) * step, (j + 1) * step);
			image1 = morphed.clone();
			inputPts = outputPts;
			imshow("morped", morphed);
			waitKey(100);
			output.write(morphed);
		}

		double alpha = 1;
		double beta = 0;
		Mat dst;
		for (size_t j = 0; j < 10; ++j) {
			beta = (1.0 - alpha);
			addWeighted(morphed, alpha, orig2, beta, 0.0, dst);
			output.write(dst);
			alpha -= 0.1;
		}

		for (size_t j = 0; j < 10; ++j) {
			output.write(orig2);
		}

		image1 = image2.clone();
	}
	return 0;
}
