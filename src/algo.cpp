#include "algo.hpp"
#include "draw.hpp"
#include "util.hpp"
#include "blend.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <algorithm>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

using namespace std;
using namespace cv;
namespace poppy {
bool show_gui = false;
double number_of_frames = 60;
size_t len_iterations = -1;
double target_len_diff = 0;
size_t ang_iterations = -1;
double target_ang_diff = 0;
double match_tolerance = 1;
double contour_sensitivity = 1;
off_t max_keypoints = -1;
size_t pyramid_levels = 4;


void canny_threshold(const Mat &src, Mat &detected_edges, double thresh) {
	detected_edges = src.clone();
	GaussianBlur(src, detected_edges, Size(9, 9), 1);
	Canny(detected_edges, detected_edges, thresh, thresh * 2);
}

void angle_test(std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols, int rows) {
	if (target_ang_diff == 0)
		return;

	std::vector<std::tuple<double, std::vector<KeyPoint>, std::vector<KeyPoint>>> diffs;
	for (size_t i = 0; i < hypot(cols, rows); ++i) {
		double avg = 0;
		double total = 0;
		for (size_t j = 0; j < kpv1.size(); ++j) {
			total += M_PI + std::atan2(kpv2[j].pt.y - kpv1[j].pt.y, (cols + kpv2[j].pt.x) - kpv1[j].pt.x);
		}

		avg = total / kpv1.size();
		double dev = avg / ((100.0 / ((i + 1.0))) * 100.0);
		double min = avg - (dev / 2.0);
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
		double score1 = 1.0 - std::abs(off_t(kpv1.size()) - off_t(kpv2.size())) / std::max(kpv1.size(), kpv2.size());
		double score2 = 1.0 - std::fabs((off_t(kpv1.size()) - off_t(new1.size())) - (kpv1.size() / (100.0 / target_ang_diff))) / (kpv1.size() / (100.0 / (100.0 - target_ang_diff)));
		double score3 = 1.0 - std::fabs((off_t(kpv2.size()) - off_t(new2.size())) - (kpv2.size() / (100.0 / target_ang_diff))) / (kpv2.size() / (100.0 / (100.0 - target_ang_diff)));
		if (score1 < 0)
			score1 = 0;

		if (score2 < 0)
			score2 = 0;

		assert(score1 <= 1.0);
		assert(score2 <= 1.0);
		diffs.push_back( { (score1 * 0.5) * score2 * score3, new1, new2 });
	}

	double score = 0;
	double maxScore = -1;
	size_t candidate = 0;

	for (size_t i = 0; i < diffs.size(); ++i) {
		auto &dt = diffs[i];
		score = std::get<0>(dt);
		if (score > maxScore) {
			maxScore = score;
			candidate = i;
		}
	}
	std::cerr << "angle test: " << std::get<1>(diffs[candidate]).size() << "/" << (((kpv1.size() - std::get<1>(diffs[candidate]).size()) / double(kpv1.size())) * 100) << std::endl;
	kpv1 = std::get<1>(diffs[candidate]);
	kpv2 = std::get<2>(diffs[candidate]);
}

void angle_test(std::vector<Point2f> &ptv1, std::vector<Point2f> &ptv2, int cols, int rows) {
	std::vector<KeyPoint> kpv1, kpv2;

	for (auto pt : ptv1)
		kpv1.push_back( { pt, 1 });

	for (auto pt : ptv2)
		kpv2.push_back( { pt, 1 });

	angle_test(kpv1, kpv2, cols, rows);
	ptv1.clear();
	ptv2.clear();

	for (auto kp : kpv1)
		ptv1.push_back(kp.pt);

	for (auto kp : kpv2)
		ptv2.push_back(kp.pt);
}

void length_test(std::vector<std::tuple<KeyPoint, KeyPoint, double>> edges, std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols, int rows) {
	if (target_len_diff == 0) {
		kpv1.clear();
		kpv2.clear();
		for (auto e : edges) {
			kpv1.push_back(std::get<0>(e));
			kpv2.push_back(std::get<1>(e));
		}
		return;
	}

	std::vector<std::tuple<double, std::vector<KeyPoint>, std::vector<KeyPoint>>> diffs;
	for (size_t i = 0; i < hypot(cols, rows); ++i) {
		double avg = 0;
		double total = 0;

		for (auto e : edges) {
			total += std::get<2>(e);
		}

		avg = total / edges.size();
		double dev = avg / ((100.0 / ((i + 1.0))) * 100.0);
		double min = avg - (dev / 2.0);
		double max = min + dev;

		std::vector<KeyPoint> new1;
		std::vector<KeyPoint> new2;

		for (auto e : edges) {
			double len = std::get<2>(e);
			if (len > min && len < max) {
				new1.push_back(std::get<0>(e));
				new2.push_back(std::get<1>(e));
			}
		}
		double score = 1.0 - std::fabs((double(edges.size()) - double(new1.size())) - (edges.size() / (100.0 / target_len_diff))) / (edges.size() / (100.0 / (100.0 - target_len_diff)));
		if (score < 0)
			score = 0;
		assert(score <= 1.0);
		diffs.push_back( { score, new1, new2 });
	}

	double score = 0;
	double maxScore = -1;
	size_t candidate = 0;

	for (size_t i = 0; i < diffs.size(); ++i) {
		auto &dt = diffs[i];
		score = std::get<0>(dt);
		if (score > maxScore) {
			maxScore = score;
			candidate = i;
		}
	}
	std::cerr << "length test: " << std::get<1>(diffs[candidate]).size() << "/" << (((edges.size() - std::get<1>(diffs[candidate]).size()) / double(edges.size())) * 100) << std::endl;
	kpv1 = std::get<1>(diffs[candidate]);
	kpv2 = std::get<2>(diffs[candidate]);
}

void length_test(std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols, int rows) {
	std::vector<std::tuple<KeyPoint, KeyPoint, double>> edges;
	edges.reserve(10000);
	Point2f p1, p2;
	for (auto &kp1 : kpv1) {
		for (auto &kp2 : kpv2) {
			edges.push_back( { kp1, kp2, distance(kp1.pt, Point2f(kp2.pt.x + cols, kp2.pt.y)) });
		}
	}

	length_test(edges, kpv1, kpv2, cols, rows);
}

void length_test(std::vector<Point2f> &ptv1, std::vector<Point2f> &ptv2, int cols, int rows) {
	std::vector<KeyPoint> kpv1, kpv2;

	for (auto pt : ptv1)
		kpv1.push_back( { pt, 1 });

	for (auto pt : ptv2)
		kpv2.push_back( { pt, 1 });

	length_test(kpv1, kpv2, cols, rows);
	ptv1.clear();
	ptv2.clear();

	for (auto kp : kpv1)
		ptv1.push_back(kp.pt);

	for (auto kp : kpv2)
		ptv2.push_back(kp.pt);
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

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
			dstPoints.push_back(pt[0]);
			dstPoints.push_back(pt[1]);
			dstPoints.push_back(pt[2]);
		}
	}
}

std::pair<std::vector<Point2f>, std::vector<Point2f>> find_matches(const Mat &grey1, const Mat &grey2) {
	if (max_keypoints == -1)
		max_keypoints = hypot(grey1.cols, grey1.rows) / 4.0;
	cv::Ptr<cv::ORB> detector = cv::ORB::create(max_keypoints);
	cv::Ptr<cv::ORB> extractor = cv::ORB::create();

	std::vector<KeyPoint> keypoints1, keypoints2;

	Mat descriptors1, descriptors2;
	detector->detect(grey1, keypoints1);
	detector->detect(grey2, keypoints2);

	detector->compute(grey1, keypoints1, descriptors1);
	detector->compute(grey2, keypoints2, descriptors2);

	Mat matMatches;

	std::vector<Point2f> points1, points2;
	for (auto pt1 : keypoints1)
		points1.push_back(pt1.pt);

	for (auto pt2 : keypoints2)
		points2.push_back(pt2.pt);

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

void morph_points(std::vector<cv::Point2f> &srcPts1, std::vector<cv::Point2f> &srcPts2, std::vector<cv::Point2f> &dstPts, float s) {
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

void draw_contour_map(std::vector<std::vector<std::vector<cv::Point>>> &collected, vector<Vec4i> &hierarchy, Mat &dst, int cols, int rows, int type) {
	dst = Mat::zeros(rows, cols, type);

	for (size_t i = 0; i < collected.size(); ++i) {
		auto &contours = collected[i];
		double shade = 0;

		for (size_t j = 0; j < contours.size(); ++j) {
			shade = 32.0 + 223.0 * (double(j) / contours.size());
			cv::drawContours(dst, contours, j, { shade, shade, shade }, 1.0, cv::LINE_8, hierarchy, 0);
		}
	}
}

void find_contours(const Mat &img1, const Mat &img2, std::vector<Mat> &dst1, std::vector<Mat> &dst2, Mat &allContours1, Mat &allContours2) {
	std::vector<std::vector<cv::Point>> contours1;
	std::vector<std::vector<cv::Point>> contours2;
	Mat median1, median2, lap1, lap2, grey1, grey2;

	vector<Vec4i> hierarchy1;
	vector<Vec4i> hierarchy2;
	cvtColor(img1, grey1, cv::COLOR_RGB2GRAY);
	cvtColor(img2, grey2, cv::COLOR_RGB2GRAY);

	medianBlur(grey1, median1, 3);
	medianBlur(grey2, median2, 3);

	Laplacian(median1, lap1, median1.depth());
	Laplacian(median2, lap2, median2.depth());
	Mat sharp1 = grey1 - (0.7 * lap1);
	Mat sharp2 = grey2 - (0.7 * lap2);
	Mat eq1;
	Mat eq2;

	equalizeHist(sharp1, eq1);
	equalizeHist(sharp2, eq2);

	Mat fgMask1;
	Mat fgMask2;

	auto pBackSub1 = createBackgroundSubtractorMOG2();
	medianBlur(eq1, median1, 3);
	pBackSub1->apply(eq1, fgMask1);
	pBackSub1->apply(median1, fgMask1);

	auto pBackSub2 = createBackgroundSubtractorMOG2();
	medianBlur(eq2, median2, 3);
	pBackSub2->apply(eq2, fgMask2);
	pBackSub2->apply(median2, fgMask2);

	eq1 = median1 * 0.25 + fgMask1 * 0.75;
	eq2 = median2 * 0.25 + fgMask2 * 0.75;
	show_image("eq1", eq1);
	show_image("eq2", eq2);

	double t1 = 0, t2 = 0;
	Mat thresh1, thresh2;
	std::vector<std::vector<std::vector<cv::Point>>> collected1;
	std::vector<std::vector<std::vector<cv::Point>>> collected2;

	for (off_t i = 0; i < 16; ++i) {
		t1 = std::max(0, std::min(255, (int) round(i * 16.0 * contour_sensitivity)));
		t2 = std::max(0, std::min(255, (int) round((i + 1) * 16.0 * contour_sensitivity)));
		cv::threshold(eq1, thresh1, t1, t2, 0);
		cv::findContours(thresh1, contours1, hierarchy1, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);
		collected1.push_back(contours1);
	}

	Mat cmap1, cmap2;
	draw_contour_map(collected1, hierarchy1, cmap1, grey1.cols, grey1.rows, grey1.type());
	show_image("cmap1", cmap1);

	size_t off = 0;
	for (off_t j = 0; j < 16; ++j) {
		t1 = std::min(255, (int) round((off + j) * 16 * contour_sensitivity));
		t2 = std::min(255, (int) round((off + j + 1) * 16 * contour_sensitivity));
		cv::threshold(eq2, thresh2, t1, t2, 0);
		cv::findContours(thresh2, contours2, hierarchy2, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);
		collected2.push_back(contours2);
	}

	draw_contour_map(collected2, hierarchy2, cmap2, thresh2.cols, thresh2.rows, thresh2.type());

	allContours1 = cmap1.clone();
	allContours2 = cmap2.clone();

	dst1.clear();
	dst2.clear();
	size_t minC = std::min(collected1.size(), collected2.size());
	dst1.resize(minC);
	dst2.resize(minC);

	for (size_t i = 0; i < minC; ++i) {
		Mat &cont1 = dst1[i];
		Mat &cont2 = dst2[i];
		cont1 = Mat::zeros(img1.rows, img1.cols, img1.type());
		cont2 = Mat::zeros(img2.rows, img2.cols, img2.type());
		contours1 = collected1[i];
		contours2 = collected2[i];

		for (size_t j = 0; j < contours1.size(); ++j) {
			cv::drawContours(cont1, contours1, j, { 255, 255, 255 }, 1.0, cv::LINE_8, hierarchy1, 0);
		}

		for (size_t j = 0; j < contours2.size(); ++j) {
			cv::drawContours(cont2, contours2, j, { 255, 255, 255 }, 1.0, cv::LINE_8, hierarchy2, 0);
		}

		cvtColor(cont1, cont1, cv::COLOR_RGB2GRAY);
		cvtColor(cont2, cont2, cv::COLOR_RGB2GRAY);
	}

	show_image("Conto1", allContours1);
	show_image("Conto2", allContours2);
}

void find_matches(Mat &orig1, Mat &orig2, std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2, Mat &allContours1, Mat &allContours2) {

	std::vector<Mat> contours1, contours2;
	find_contours(orig1, orig2, contours1, contours2, allContours1, allContours2);

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
//	draw_matches(grey1, grey2, matMatches, srcPoints1, srcPoints2);
//	if(show_gui) imshow("matches", matMatches);
}

std::tuple<double, double, double> calculate_sum_mean_and_sd(std::multimap<double, std::pair<Point2f, Point2f>> distanceMap) {
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

void match_points_by_proximity(std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2, int cols, int rows) {
	std::multimap<double, std::pair<Point2f, Point2f>> distanceMap;

	Point2f nopoint(-1, -1);
	for (auto &pt1 : srcPoints1) {
		double dist = 0;
		double currentMinDist = std::numeric_limits<double>::max();

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
	srcPoints1.clear();
	srcPoints2.clear();
	assert(!distanceMap.empty());
	assert(std::get<1>(distribution) != 0 && std::get<2>(distribution) != 0);
	double distance = (*distanceMap.rbegin()).first;
	double mean = std::get<1>(distribution);
	double sd = std::get<2>(distribution);
	double highZScore = std::fabs((distance - mean) / sd);
	double zScore = 0;
	double value = 0;
	double limit = match_tolerance * highZScore;

	for (auto it = distanceMap.rbegin(); it != distanceMap.rend(); ++it) {
		value = (*it).first;
		zScore = std::fabs((value - mean) / sd);
//		std::cerr
//				<< "\tm/s/l/z/h: " << mean << "/" << sd << "/" << limit << "/" << zScore << "/" << highZScore
//				<< std::endl;

		if (value < mean && zScore < limit) {
			srcPoints1.push_back((*it).second.first);
			srcPoints2.push_back((*it).second.second);
		}
	}

	assert(srcPoints1.size() == srcPoints2.size());
	check_points(srcPoints1, cols, rows);
	check_points(srcPoints2, cols, rows);
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

void prepare_matches(Mat &origImg1, Mat &origImg2, const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2) {
	//edit matches
	std::cerr << "prepare: " << srcPoints1.size() << " -> ";
	match_points_by_proximity(srcPoints1, srcPoints2, img1.cols, img1.rows);
	std::cerr << "match: " << srcPoints1.size() << " -> ";

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
	length_test(edges, kpv1, kpv2, img1.cols, img1.rows);

	srcPoints1.clear();
	for (auto kp : kpv1) {
		srcPoints1.push_back(kp.pt);
	}
	srcPoints2.clear();
	for (auto kp : kpv2) {
		srcPoints2.push_back(kp.pt);
	}
	std::cerr << "length test: " << srcPoints1.size() << std::endl;

	angle_test(srcPoints1, srcPoints2, img1.cols, img1.rows);

	Mat matMatches;
	Mat grey1, grey2;
	cvtColor(origImg1, grey1, cv::COLOR_RGB2GRAY);
	cvtColor(origImg2, grey2, cv::COLOR_RGB2GRAY);
	draw_matches(grey1, grey2, matMatches, srcPoints1, srcPoints2);
	show_image("matches reduced", matMatches);

	if (srcPoints1.size() > srcPoints2.size())
		srcPoints1.resize(srcPoints2.size());
	else
		srcPoints2.resize(srcPoints1.size());

	add_corners(srcPoints1, srcPoints2, origImg1.size);
}

double morph_images(const Mat &origImg1, const Mat &origImg2, cv::Mat &dst, const cv::Mat &last, std::vector<cv::Point2f> &morphedPoints, std::vector<cv::Point2f> srcPoints1, std::vector<cv::Point2f> srcPoints2, Mat &allContours1, Mat &allContours2, double shapeRatio, double maskRatio) {
	cv::Size SourceImgSize(origImg1.cols, origImg1.rows);
	cv::Subdiv2D subDiv1(cv::Rect(0, 0, SourceImgSize.width, SourceImgSize.height));
	cv::Subdiv2D subDiv2(cv::Rect(0, 0, SourceImgSize.width, SourceImgSize.height));
	cv::Subdiv2D subDivMorph(cv::Rect(0, 0, SourceImgSize.width, SourceImgSize.height));

	std::vector<cv::Point2f> uniq1, uniq2, uniqMorph;
	check_points(srcPoints1, origImg1.cols, origImg1.rows);
	make_uniq(srcPoints1, uniq1);
	check_uniq(uniq1);
	for (auto pt : uniq1)
		subDiv1.insert(pt);

	check_points(srcPoints2, origImg2.cols, origImg2.rows);
	make_uniq(srcPoints2, uniq2);
	check_uniq(uniq2);
	for (auto pt : uniq2)
		subDiv2.insert(pt);

	morph_points(srcPoints1, srcPoints2, morphedPoints, shapeRatio);
	assert(srcPoints1.size() == srcPoints2.size() && srcPoints2.size() == morphedPoints.size());

	check_points(morphedPoints, origImg1.cols, origImg1.rows);
	make_uniq(morphedPoints, uniqMorph);
	check_uniq(uniqMorph);
	for (auto pt : uniqMorph)
		subDivMorph.insert(pt);

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

	Mat_<Vec3f> l;
	Mat_<Vec3f> r;
	trImg1.convertTo(l, CV_32F, 1.0 / 255.0);
	trImg2.convertTo(r, CV_32F, 1.0 / 255.0);
	Mat_<float> m(l.rows, l.cols, 0.0);
	Mat_<float> m1(l.rows, l.cols, 0.0);
	Mat_<float> m2(l.rows, l.cols, 0.0);
	equalizeHist(allContours1, allContours1);
	equalizeHist(allContours1, allContours1);

	for (off_t x = 0; x < m1.cols; ++x) {
		for (off_t y = 0; y < m1.rows; ++y) {
			m2.at<float>(y, x) = allContours1.at<uint8_t>(y, x) / 255.0;
		}
	}

	for (off_t x = 0; x < m2.cols; ++x) {
		for (off_t y = 0; y < m2.rows; ++y) {
			m1.at<float>(y, x) = allContours2.at<uint8_t>(y, x) / 255.0;
		}
	}

	m2(Range::all(), Range::all()) = 1.0;
	Mat mask;
	m = (m2 * (1.0 - maskRatio) + m1 * maskRatio);

	off_t kx = std::round(m.cols / 32.0);
	off_t ky = std::round(m.rows / 32.0);
	if (kx % 2 != 1)
		kx -= 1;

	if (ky % 2 != 1)
		ky -= 1;

	GaussianBlur(m, mask, Size(kx, ky), 12);
	LaplacianBlending lb(l, r, mask, pyramid_levels);
	Mat_<Vec3f> lapBlend = lb.blend().clone();
	lapBlend.convertTo(dst, origImg1.depth(), 255.0);
	Mat analysis = dst.clone();
	Mat prev = last.clone();
	if (prev.empty())
		prev = dst.clone();
	draw_morph_analysis(dst, prev, analysis, SourceImgSize, subDiv1, subDiv2, subDivMorph, { 0, 0, 255 });
	show_image("analysis", analysis);
	return 0;
}
}
