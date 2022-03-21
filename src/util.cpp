#include "util.hpp"
#include "settings.hpp"


#include <opencv2/highgui/highgui.hpp>
#include <set>

using namespace std;
using namespace cv;

namespace poppy {
void show_image(const string &name, const Mat &img) {
	if(Settings::instance().show_gui) {
		namedWindow(name, WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
		imshow(name, img);
	}
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
		assert(pt.x < cols && pt.y < rows);
	}
#endif
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
