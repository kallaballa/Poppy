#include "util.hpp"
#include "settings.hpp"


#include <opencv2/highgui/highgui.hpp>
#include <set>
#include <iostream>

using namespace std;
using namespace cv;

namespace poppy {

void overdefineHull(vector<Point2f>& hull, size_t minPoints) {
	assert(hull.size() > 1);
	off_t diff = minPoints - hull.size();
	while(diff > 0)  {
		for(size_t i = 0; i < hull.size() && diff > 0; i+=2) {
			const auto& first = hull[i];
			const auto& second = hull[(i + 1) % hull.size()];
			auto insertee = first;
			auto vector = second - first;
			vector.x /= 2.0;
			vector.y /= 2.0;
			insertee.x += vector.x;
			insertee.y += vector.y;
			hull.insert(hull.begin() + (i+1), std::move(insertee));
			--diff;
		}
	}
}

double morph_distance(const vector<Point2f>& srcPoints1, const vector<Point2f>& srcPoints2, const double width, const double height) {
	assert(srcPoints1.size() == srcPoints2.size());
	double totalDistance = 0;
	for(size_t i = 0; i < srcPoints1.size(); ++i) {
		Point2f v = srcPoints2[i] - srcPoints1[i];
		totalDistance += hypot(v.x, v.y);
	}
	return totalDistance / srcPoints1.size();
}

void show_image(const string &name, const Mat &img) {
#ifndef _WASM
	if(Settings::instance().show_gui) {
		namedWindow(name, WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
		imshow(name, img);
	}
#endif
}

void wait_key(int timeout) {
#ifndef _WASM
	if(Settings::instance().show_gui) {
		waitKey(timeout);
	}
#endif
}

void clip_points(std::vector<Point2f> &pts, int cols, int rows) {
	for (auto &pt : pts) {
		pt.x = pt.x > cols ? cols - 1 : pt.x;
		pt.y = pt.y > rows ? rows - 1 : pt.y;
		pt.x = pt.x < 0 ? 0 : pt.x;
		pt.y = pt.y < 0 ? 0 : pt.y;
	}
}

void check_points(const std::vector<Point2f> &pts, int cols, int rows) {
#ifndef NDEBUG
	for (const auto &pt : pts) {
		assert(!isinf(pt.x) && !isinf(pt.y));
		assert(!isnan(pt.x) && !isnan(pt.y));
		assert(pt.x >= 0 && pt.y >= 0);
		assert(pt.x <= cols && pt.y <= rows);
	}
#endif
}

void filter_invalid_points(vector<Point2f>& srcPoints1, vector<Point2f>& srcPoints2, int cols, int rows) {
	for (size_t i = 0; i < srcPoints1.size(); ++i) {
		Point2f& pt = srcPoints1[i];
		if(pt.x < 0 || pt.x > cols || pt.y < 0 || pt.y > rows) {
			srcPoints1.erase(srcPoints1.begin()+i);
			srcPoints2.erase(srcPoints2.begin()+i);
			--i;
		}
	}
	for (size_t i = 0; i < srcPoints2.size(); ++i) {
		Point2f& pt = srcPoints2[i];
		if(pt.x < 0 || pt.x > cols || pt.y < 0 || pt.y > rows) {
			srcPoints1.erase(srcPoints1.begin()+i);
			srcPoints2.erase(srcPoints2.begin()+i);
			--i;
		}
	}
}


void check_uniq(const std::vector<Point2f> &pts) {
#ifndef NDEBUG
	std::set<Point2f, LessPointOp> uniq;
	for (const auto &pt : pts) {
		assert(uniq.insert(pt).second);
	}
#endif
}

void check_min_distance(const std::vector<Point2f> &in, double minDistance) {
#ifndef NDEBUG
	double dist = 0;
	for (size_t i = 0; i < in.size(); ++i) {
		const auto& pt1 = in[i];
		for (size_t j = 0; j < in.size(); ++j) {
			if(i == j)
				continue;
			const auto& pt2 = in[j];
			dist = hypot(pt2.x - pt1.x, pt2.y - pt1.y);
			assert(dist >= minDistance);
		}
	}
#endif
}

void filter_min_distance(const std::vector<Point2f> &in, std::vector<Point2f> &out, double minDistance) {
	double dist = 0;
	for (size_t i = 0; i < in.size(); ++i) {
		const auto& pt1 = in[i];
		out.push_back(pt1);
		for (size_t j = i + 1; j < in.size(); ++j) {
			assert(i != j);
			const auto& pt2 = in[j];
			dist = hypot(pt2.x - pt1.x, pt2.y - pt1.y);
			if(dist >= minDistance) {
				out.push_back(pt2);
			}
		}
	}
}

void make_uniq(const std::vector<Point2f> &pts, std::vector<Point2f> &out) {
	std::set<Point2f, LessPointOp> uniq;
	for (const auto &pt : pts) {
		if (uniq.insert(pt).second) {
			out.push_back(pt);
		}
	}
}

bool operator==(const Point2f &pt1, const Point2f &pt2) {
	return pt1.x == pt2.x && pt1.y == pt2.y;
}

bool operator==(const KeyPoint &kp1, const KeyPoint &kp2) {
	return kp1.pt.x == kp2.pt.x && kp1.pt.y == kp2.pt.y;
}

double distance(const Point2f &p1, const Point2f &p2) {
	return hypot(p2.x - p1.x, p2.y - p1.y);
}
}
