#ifndef SRC_UTIL_HPP_
#define SRC_UTIL_HPP_

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>

namespace poppy {
struct LessPointOp {
	bool operator()(const cv::Point2f &lhs, const cv::Point2f &rhs) const {
		return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
	}
};

void show_image(const std::string &name, const cv::Mat &img);
void check_points(const std::vector<cv::Point2f> &pts, int cols, int rows);
void check_uniq(const std::vector<cv::Point2f> &pts);
void check_min_distance(const std::vector<cv::Point2f> &in, double minDistance);
void filter_min_distance(const std::vector<cv::Point2f> &in, std::vector<cv::Point2f> &out, double minDistance);
void make_uniq(const std::vector<cv::Point2f> &pts, std::vector<cv::Point2f> &out);
bool operator==(const cv::Point2f &pt1, const cv::Point2f &pt2);
bool operator==(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2);
double distance(const cv::Point2f &p1, const cv::Point2f &p2);
}
#endif
