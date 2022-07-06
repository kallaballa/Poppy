#ifndef SRC_UTIL_HPP_
#define SRC_UTIL_HPP_

#include <vector>

#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <map>
#include <tuple>

using std::pair;
using std::vector;
using namespace cv;

namespace cv
{
bool operator<(Point2f const& lhs, Point2f const& rhs);
}

namespace poppy {

constexpr double RADIANS_TO_DEGREES = 57.2958;

template<typename T>
struct reversion_wrapper {
	T &iterable;
};

template<typename T>
auto begin(reversion_wrapper<T> w) {
	return std::rbegin(w.iterable);
}

template<typename T>
auto end(reversion_wrapper<T> w) {
	return std::rend(w.iterable);
}

template<typename T>
reversion_wrapper<T> reverse(T &&iterable) {
	return {iterable};
}

template<typename C>
double meanAngle(const C& c) {
    auto it = std::cbegin(c);
    auto end = std::cend(c);

    double x = 0.0;
    double y = 0.0;
    double len = 0.0;
    while (it != end) {
        x += cos(*it);
        y += sin(*it);
        len++;

        it = std::next(it);
    }

    return atan2(y, x);
}

template<typename C>
double weightedAngle(const C& c) {
    auto it = std::cbegin(c);
    auto end = std::cend(c);

    double x = 0.0;
    double y = 0.0;
    double len = 0.0;
    while (it != end) {
        x += cos((*it).first) * (*it).second;
        y += sin((*it).first) * (*it).second;
        len++;

        it = std::next(it);
    }

    return atan2(y, x);
}

void color_reduce(cv::Mat& image, int div);
struct LessPointOp {
	bool operator()(const cv::Point2f &lhs, const cv::Point2f &rhs) const {
		return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
	}
};
void triple_channel(const Mat& src, Mat& dst);
Mat unsharp_mask(const Mat& original, float radius, float amount, float threshold);
double feature_metric(const Mat &grey1);
std::tuple<double, double, double> calculate_sum_mean_and_sd(std::multimap<double, pair<Point2f, Point2f>> distanceMap);
void add_corners(vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, cv::MatSize sz);
void normalize(const Mat &src, Mat &dst);
void adjust_contrast_and_brightness(const cv::Mat &src, cv::Mat &dst, double contrast, double lowcut);
std::vector<std::vector<cv::Point2f>> convertContourTo2f(const std::vector<std::vector<cv::Point>> &contours1);
std::vector<std::vector<cv::Point>> convertContourFrom2f(const std::vector<std::vector<cv::Point2f>> &contours1);
void overdefineHull(std::vector<cv::Point2f>& hull, size_t minPoints);
double morph_distance(const std::vector<cv::Point2f>& srcPoints1, const std::vector<cv::Point2f>& srcPoints2, const double& width, const double& height);
void show_image(const std::string &name, const cv::Mat &img);
void wait_key(int timeout = 0);
void clip_points(std::vector<cv::Point2f> &pts, int cols, int rows);
void check_points(const std::vector<cv::Point2f> &pts, int cols, int rows);
void filter_invalid_points(std::vector<cv::Point2f>& srcPoints1, std::vector<cv::Point2f>& srcPoints2, int cols, int rows);
void check_uniq(const std::vector<cv::Point2f> &pts);
void check_min_distance(const std::vector<cv::Point2f> &in, double minDistance);
void filter_min_distance(const std::vector<cv::Point2f> &in, std::vector<cv::Point2f> &out, double minDistance);
void make_uniq(const std::vector<cv::Point2f> &pts, std::vector<cv::Point2f> &out);
bool operator==(const cv::Point2f &pt1, const cv::Point2f &pt2);
bool operator==(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2);
double distance(const cv::Point2f &p1, const cv::Point2f &p2);
cv::Point2f average(const std::vector<cv::Point2f> &pts);
}
#endif
