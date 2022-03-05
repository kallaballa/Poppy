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

#ifndef _WASM
#include <boost/program_options.hpp>
#endif

#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

bool show_gui = false;
double number_of_frames = 60;
size_t len_iterations = 2000;
double target_len_diff = 10;
size_t ang_iterations = 2000;
double target_ang_diff = 5;
double match_sensitivity = 1.0;
double contour_sensitivity = 0.5;
off_t max_keypoints = -1;
size_t pyramid_levels = 4;

#ifndef _WASM
namespace po = boost::program_options;
#endif

typedef unsigned char sample_t;

using namespace std;
using namespace cv;

class LaplacianBlending {
private:
	Mat_<Vec3f> left;
	Mat_<Vec3f> right;
	Mat_<float> blendMask;
	vector<Mat_<Vec3f> > leftLapPyr, rightLapPyr, resultLapPyr;
	Mat leftSmallestLevel, rightSmallestLevel, resultSmallestLevel;
	vector<Mat_<Vec3f> > maskGaussianPyramid; //masks are 3-channels for easier multiplication with RGB
	int levels;
	void buildPyramids() {
		buildLaplacianPyramid(left, leftLapPyr, leftSmallestLevel);
		buildLaplacianPyramid(right, rightLapPyr, rightSmallestLevel);
		buildGaussianPyramid();
	}
	void buildGaussianPyramid() {
		assert(leftLapPyr.size() > 0);
		maskGaussianPyramid.clear();
		Mat currentImg;
		cvtColor(blendMask, currentImg, COLOR_GRAY2BGR);
		maskGaussianPyramid.push_back(currentImg); //highest level
		currentImg = blendMask;
		for (int l = 1; l < levels + 1; l++) {
			Mat _down;
			if (off_t(leftLapPyr.size()) > l) {
				pyrDown(currentImg, _down, leftLapPyr[l].size());
			} else {
				pyrDown(currentImg, _down, leftSmallestLevel.size()); //smallest level
			}
			Mat down;
			cvtColor(_down, down, COLOR_GRAY2BGR);
			maskGaussianPyramid.push_back(down);
			currentImg = _down;
		}
	}
	void buildLaplacianPyramid(const Mat &img, vector<Mat_<Vec3f> > &lapPyr, Mat &smallestLevel) {
		lapPyr.clear();
		Mat currentImg = img;
		for (int l = 0; l < levels; l++) {
			Mat down, up;
			pyrDown(currentImg, down);
			pyrUp(down, up, currentImg.size());
			Mat lap = currentImg - up;
			lapPyr.push_back(lap);
			currentImg = down;
		}
		currentImg.copyTo(smallestLevel);
	}
	Mat_<Vec3f> reconstructImgFromLapPyramid() {
		Mat currentImg = resultSmallestLevel;
		for (int l = levels - 1; l >= 0; l--) {
			Mat up;
			pyrUp(currentImg, up, resultLapPyr[l].size());
			currentImg = up + resultLapPyr[l];
		}
		return currentImg;
	}
	void blendLapPyrs() {
		resultSmallestLevel = leftSmallestLevel.mul(maskGaussianPyramid.back()) +
				rightSmallestLevel.mul(Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid.back());
		for (int l = 0; l < levels; l++) {
			Mat A = leftLapPyr[l].mul(maskGaussianPyramid[l]);
			Mat antiMask = Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid[l];
			Mat B = rightLapPyr[l].mul(antiMask);
			Mat_<Vec3f> blendedLevel = A + B;
			resultLapPyr.push_back(blendedLevel);
		}
	}
public:
	LaplacianBlending(const Mat_<Vec3f> &_left, const Mat_<Vec3f> &_right, const Mat_<float> &_blendMask, int _levels) :
			left(_left), right(_right), blendMask(_blendMask), levels(_levels)
	{
		assert(_left.size() == _right.size());
		assert(_left.size() == _blendMask.size());
		buildPyramids();
		blendLapPyrs();
	}

	Mat_<Vec3f> blend() {
		return reconstructImgFromLapPyramid();
	}
};
Mat_<Vec3f> LaplacianBlend(const Mat_<Vec3f> &l, const Mat_<Vec3f> &r, const Mat_<float> &m) {
	LaplacianBlending lb(l, r, m, 4);
	return lb.blend();
}

void show_image(const string &name, const Mat &img) {
	if (show_gui) {
		namedWindow(name, WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
		imshow(name, img);
	}
}

void check_points(const std::vector<Point2f> &pts, int cols, int rows) {
	for (const auto &pt : pts) {
		assert(!isinf(pt.x) && !isinf(pt.y));
		assert(!isnan(pt.x) && !isnan(pt.y));
		assert(pt.x >= 0 && pt.y >= 0);
		assert(pt.x < cols && pt.y < rows);
	}
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
	Canny(detected_edges, detected_edges, thresh, thresh * 2);
}

void saturate(const cv::Mat &img, cv::Mat &saturated, double changeBy) {
	Mat imgHsv;
	cvtColor(img, imgHsv, COLOR_RGB2HSV);

	for (int y = 0; y < imgHsv.cols; y++) {
		for (int x = 0; x < imgHsv.rows; x++) {
			Vec3b &pix = imgHsv.at<Vec3b>(Point(y, x));
			int s = pix[1];
			s += changeBy;
			if (s < 0)
				s = 0;
			else if (s > 255)
				s = 255;
			pix[1] = s;
		}
	}

	cvtColor(imgHsv, saturated, COLOR_HSV2RGB);
}

void angle_test(std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols) {
	std::vector<std::tuple<double, std::vector<KeyPoint>, std::vector<KeyPoint>>> diffs;
	for (size_t i = 0; i < ang_iterations; ++i) {
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
		assert(score1 >= 0 && score2 >= 0);
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

void angle_test(std::vector<Point2f> &ptv1, std::vector<Point2f> &ptv2, int cols) {
	std::vector<KeyPoint> kpv1, kpv2;

	for (auto pt : ptv1)
		kpv1.push_back( { pt, 1 });

	for (auto pt : ptv2)
		kpv2.push_back( { pt, 1 });

	angle_test(kpv1, kpv2, cols);
	ptv1.clear();
	ptv2.clear();

	for (auto kp : kpv1)
		ptv1.push_back(kp.pt);

	for (auto kp : kpv2)
		ptv2.push_back(kp.pt);
}

void length_test(std::vector<std::tuple<KeyPoint, KeyPoint, double>> edges, std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols) {
	std::vector<std::tuple<double, std::vector<KeyPoint>, std::vector<KeyPoint>>> diffs;
	for (size_t i = 0; i < len_iterations; ++i) {
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

		double score = 1.0 - std::fabs((off_t(edges.size()) - off_t(new1.size())) - (edges.size() / (100.0 / target_len_diff))) / (edges.size() / (100.0 / (100.0 - target_len_diff)));
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
	std::vector<KeyPoint> kpv1, kpv2;

	for (auto pt : ptv1)
		kpv1.push_back( { pt, 1 });

	for (auto pt : ptv2)
		kpv2.push_back( { pt, 1 });

	length_test(kpv1, kpv2, cols);
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
	Mat norm;
	normalize(flow, norm, 1.0, 0.0, NORM_MINMAX);

	for (off_t x = 0; x < morphed.cols; ++x) {
		for (off_t y = 0; y < morphed.rows; ++y) {
			circle(dst, Point(x, y), 1, Scalar(0), -1);
			const Point2f flv1 = norm.at<Point2f>(y, x);
			double mag = hypot(flv1.x, flv1.y);
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

void collect_flow_centers(const Mat &morphed, const Mat &last, std::vector<std::pair<Point2f, double>> &highlightCenters) {
	Mat flowm;
	Mat grey;
	draw_flow_heightmap(morphed, last, flowm);
	cvtColor(flowm, grey, cv::COLOR_RGB2GRAY);

	Mat overlay;
	Mat thresh;
	normalize(grey, overlay, 255.0, 0.0, NORM_MINMAX);
//	GaussianBlur(overlay, overlay, Size(13, 13), 2);
	cv::threshold(overlay, thresh, 200, 255, 0);
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
			highlightCenters.push_back( { pt, hypot(br.width, br.height) });
		}
	}
}

static std::vector<std::pair<Point2f, double>> highlights;

void draw_morph_analysis(const Mat &morphed, const Mat &last, Mat &dst, const Size &size, Subdiv2D &subdiv1, Subdiv2D &subdiv2, Subdiv2D &subdivMorph, Scalar delaunay_color) {
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

//	for (auto &h : highlights) {
//		circle(dst, h.first, 1, Scalar(255, 255, 255), -1);
//	}
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

void find_contours(const Mat &img1, const Mat &img2, std::vector<Mat> &dst1, std::vector<Mat> &dst2, Mat &allContours1, Mat &allContours2) {
	std::vector<std::vector<cv::Point>> contours1;
	std::vector<std::vector<cv::Point>> contours2;
	Mat sat1, sat2, contrast1, contrast2, blur1, blur2, grey1, grey2, thresh1, thresh2;
	vector<Vec4i> hierarchy1;
	vector<Vec4i> hierarchy2;
	saturate(img1, sat1, 255.0);
	saturate(img2, sat2, 255.0);
	sat1.convertTo(contrast1, -1, 1.2, 0);
	sat2.convertTo(contrast2, -1, 1.2, 0);
	GaussianBlur(contrast1, blur1, Size(13, 13), 2);
	GaussianBlur(contrast2, blur2, Size(13, 13), 2);
//	show_image("enh1", blur1);
//	show_image("enh2", blur2);
	cvtColor(blur1, grey1, cv::COLOR_RGB2GRAY);
	cvtColor(blur2, grey2, cv::COLOR_RGB2GRAY);

	std::vector<std::vector<std::vector<cv::Point>>> collected1;
	std::vector<std::vector<std::vector<cv::Point>>> collected2;

	for (off_t i = 0; i < 16; ++i) {
		cv::threshold(grey1, thresh1, std::min(255, (int) round((i + 1) * 16 * contour_sensitivity)), 255, 0);
		cv::findContours(thresh1, contours1, hierarchy1, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);
		collected1.push_back(contours1);
	}

	for (off_t i = 0; i < 16; ++i) {
		cv::threshold(grey2, thresh2, std::min(255, (int) round((i + 1) * 16 * contour_sensitivity)), 255, 0);
		cv::findContours(thresh2, contours2, hierarchy2, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);
		collected2.push_back(contours2);
	}

	dst1.clear();
	dst2.clear();
	dst1.resize(collected1.size());
	dst2.resize(collected1.size());

	allContours1 = Mat::zeros(grey1.rows, grey1.cols, grey1.type());
	allContours2 = Mat::zeros(grey1.rows, grey1.cols, grey1.type());

	for (size_t i = 0; i < collected1.size(); ++i) {
		Mat &cont1 = dst1[i];
		Mat &cont2 = dst2[i];
		cvtColor(thresh1, cont1, cv::COLOR_GRAY2RGB);
		cvtColor(thresh2, cont2, cv::COLOR_GRAY2RGB);
		contours1 = collected1[i];
		contours2 = collected2[i];
		double shade = 0;

		for (size_t j = 0; j < contours1.size(); ++j) {
			shade = 32.0 + 223.0 * (double(j) / contours1.size());
			cv::drawContours(cont1, contours1, j, { 255, 255, 255 }, 1, cv::LINE_8, hierarchy1, 0);
			cv::drawContours(allContours1, contours1, j, { shade, shade, shade }, 1, cv::LINE_8, hierarchy1, 0);
		}

		for (size_t j = 0; j < contours2.size(); ++j) {
			shade = 32.0 + 223.0 * (double(j) / contours1.size());
			cv::drawContours(cont2, contours2, j, { 255, 255, 255 }, 1, cv::LINE_8, hierarchy2, 0);
			cv::drawContours(allContours2, contours2, j, { shade, shade, shade }, 1, cv::LINE_8, hierarchy2, 0);
		}

		cvtColor(cont1, cont1, cv::COLOR_RGB2GRAY);
		cvtColor(cont2, cont2, cv::COLOR_RGB2GRAY);
	}

	show_image("Contours1", allContours1);
	show_image("Contours2", allContours2);
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

std::tuple<double, double, double> calculate_sum_mean_and_sd(std::map<double, std::pair<Point2f, Point2f>> distanceMap) {
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
	std::map<double, std::pair<Point2f, Point2f>> distanceMap;
	std::set<Point2f, LessPointOp> setpt2;
	for (auto pt2 : srcPoints2) {
		setpt2.insert(pt2);
	}
	for (auto &pt1 : srcPoints1) {
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
		distanceMap[dist] = { pt1, closest };
		setpt2.erase(closest);
	}
	auto distribution = calculate_sum_mean_and_sd(distanceMap);

	srcPoints1.clear();
	srcPoints2.clear();
	assert(!distanceMap.empty());

	double highZScore = ((*distanceMap.begin()).first - std::get<1>(distribution)) / std::get<2>(distribution);
	double zScore = 0;
	double factor = 0.85 * (1.0 / match_sensitivity);
	double limit = highZScore * factor;
	for (auto it = distanceMap.rbegin(); it != distanceMap.rend(); ++it) {
		zScore = ((*it).first - std::get<1>(distribution)) / std::get<2>(distribution);
		if (zScore < limit) {
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

	check_points(srcPoints1, origImg1.cols, origImg1.rows);
	for (auto pt : srcPoints1)
		subDiv1.insert(pt);

	check_points(srcPoints2, origImg1.cols, origImg1.rows);
	for (auto pt : srcPoints2)
		subDiv2.insert(pt);

	morph_points(srcPoints1, srcPoints2, morphedPoints, shapeRatio);
	assert(srcPoints1.size() == srcPoints2.size() && srcPoints2.size() == morphedPoints.size());

	check_points(morphedPoints, origImg1.cols, origImg1.rows);
	for (auto pt : morphedPoints)
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

	show_image("mask", mask);

	LaplacianBlending lb(l, r, mask, pyramid_levels);
	Mat_<Vec3f> lapBlend = lb.blend().clone();
//	Mat linearBlend = trImg1 * (1.0 - colorRatio) + trImg2 * colorRatio;
	lapBlend.convertTo(dst, origImg1.depth(), 255.0);

//	add(linearBlend * 0.5,lapBlend * 0.5, blend, noArray(), origImg2.type());

	Mat analysis = dst.clone();
	Mat prev = last.clone();
	if (prev.empty())
		prev = dst.clone();
	draw_morph_analysis(dst, prev, analysis, SourceImgSize, subDiv1, subDiv2, subDivMorph, { 0, 0, 255 });
	show_image("analysis", analysis);
	return 0;
}

int main(int argc, char **argv) {
	using std::string;
	srand(time(NULL));
	bool showGui = show_gui;
	size_t numberOfFrames = number_of_frames;
	double matchSensitivity = match_sensitivity;
	double targetAngDiff = target_ang_diff;
	double targetLenDiff = target_len_diff;
	double contourSensitivity = contour_sensitivity;
	off_t maxKeypoints = max_keypoints;
	std::vector<string> imageFiles;
	string outputFile = "output.mkv";
#ifndef _WASM
	po::options_description genericDesc("Options");
	genericDesc.add_options()
	("gui,g", "Show analysis windows")
	("maxkey,m", po::value<off_t>(&maxKeypoints)->default_value(maxKeypoints), "Manual override for the number of keypoints to retain during detection. The default is automatic determination of that number")
	("frames,f", po::value<size_t>(&numberOfFrames)->default_value(numberOfFrames), "The number of frames to generate")
	("sensitivity,s", po::value<double>(&matchSensitivity)->default_value(matchSensitivity), "How tolerant poppy is when matching keypoints.")
	("angloss,a", po::value<double>(&targetAngDiff)->default_value(targetAngDiff), "The target loss, in percent, for the angle test. The default is probably fine.")
	("lenloss,l", po::value<double>(&targetLenDiff)->default_value(targetLenDiff), "The target loss, in percent, for the length test. The default is probably fine.")
	("contour,c", po::value<double>(&contourSensitivity)->default_value(contourSensitivity), "How sensitive poppy is to contours. Values below 1.0 reduce the sensitivity")
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
		std::cerr << "Default options will work fine on good source material. Please," << std::endl;
		std::cerr << "always make sure images are scaled and rotated to match each" << std::endl;
		std::cerr << "other. Anyway, there are a couple of options you can specifiy." << std::endl;
		std::cerr << "But usually you would only want to do this if you either have" << std::endl;
		std::cerr << "bad source material, feel like experimenting or are trying to" << std::endl;
		std::cerr << "do something funny." << std::endl;
		std::cerr << visible;
		return 0;
	}

	if (vm.count("gui")) {
		showGui = true;
	}
#endif
	for (auto p : imageFiles) {
		if (!std::filesystem::exists(p))
			throw std::runtime_error("File doesn't exist: " + p);
	}

	show_gui = showGui;
	number_of_frames = numberOfFrames;
	match_sensitivity = matchSensitivity;
	max_keypoints = maxKeypoints;
	target_ang_diff = targetAngDiff;
	target_len_diff = targetLenDiff;
	contour_sensitivity = contourSensitivity;
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

			if (image1.cols != image2.cols || image1.rows != image2.rows) {
				std::cerr << "Image file sizes don't match: " << imageFiles[i] << std::endl;
				exit(3);
			}
		} catch (...) {
			std::cerr << "Can't read (invalid?) image file: " << imageFiles[i] << std::endl;
			exit(2);
		}

		orig1 = image1.clone();
		orig2 = image2.clone();
		Mat morphed;

		std::vector<Point2f> srcPoints1, srcPoints2, morphedPoints, lastMorphedPoints;
		std::cerr << "matching: " << imageFiles[i - 1] << " -> " << imageFiles[i] << " ..." << std::endl;
		Mat allContours1, allContours2;
		find_matches(orig1, orig2, srcPoints1, srcPoints2, allContours1, allContours2);
		prepare_matches(orig1, orig2, image1, image2, srcPoints1, srcPoints2);

		float step = 1.0 / number_of_frames;
		double linear = 0;
		double shape = 0;
		double mask = 0;

		for (size_t j = 0; j < number_of_frames; ++j) {
			if (!lastMorphedPoints.empty())
				srcPoints1 = lastMorphedPoints;
			morphedPoints.clear();

			linear = j * step;
			shape = ((1.0 / (1.0 - linear)) / number_of_frames);
			mask = shape;
			if (shape > 1.0)
				shape = 1.0;

			morph_images(image1, orig2, morphed, morphed.clone(), morphedPoints, srcPoints1, srcPoints2, allContours1, allContours2, shape, mask);
			image1 = morphed.clone();
			lastMorphedPoints = morphedPoints;
			output.write(morphed);

			show_image("morphed", morphed);
			if (show_gui)
				waitKey(1);

			std::cerr << int((j / double(number_of_frames)) * 100.0) << "%\r";
		}
		morphed.release();
		srcPoints1.clear();
		srcPoints2.clear();

		image1 = image2.clone();
	}
	return 0;
}
