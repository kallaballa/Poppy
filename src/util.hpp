#ifndef SRC_UTIL_HPP_
#define SRC_UTIL_HPP_

#include <vector>
#include <tuple>
#include <map>

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
bool rect_contains(const Rect& r, const Point2f& pt, int radius);
double euclidean_distance(cv::Point center, cv::Point point);
void gabor_filter(const Mat& src, Mat& dst, size_t numAngles = 16, int kernel_size = 13, double sig = 5, double lm = 10, double gm = 0.04, double ps = CV_PI/4);
void triple_channel(const Mat& src, Mat& dst);
void equalize_bgr(const Mat &src, Mat &dst);
Mat unsharp_mask(const Mat& original, float radius, float amount, float threshold);
double feature_metric(const Mat &grey1);
void binaries_mat(const Mat& src, Mat& dst);
pair<double, Point2f> get_orientation(const vector<Point2f> &pts);
std::tuple<vector<Point2d>, vector<double>, Point2f> get_eigen(const vector<KeyPoint> &kpts);
pair<double, Point2f> get_orientation(const vector<KeyPoint> &kpts);
std::tuple<double, double, double> calculate_sum_mean_and_sd(std::multimap<double, pair<Point2f, Point2f>> distanceMap);
void add_corners(vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, cv::MatSize sz);
void normalize(const Mat &src, Mat &dst);
void auto_adjust_contrast_and_brightness(const cv::Mat &src, cv::Mat &dst, double contrast);
std::vector<Point2f> convertPointTo2f(const std::vector<Point>& pts);
std::vector<Point> convert2fToPoint(const std::vector<Point2f>& pts);
std::vector<std::vector<cv::Point2f>> convertContourTo2f(const std::vector<std::vector<cv::Point>> &contours1);
std::vector<std::vector<cv::Point>> convertContourFrom2f(const std::vector<std::vector<cv::Point2f>> &contours1);
void overdefineHull(std::vector<cv::Point2f>& hull, size_t minPoints);
pair<vector<Point2f>, vector<Point2f>> extract_points(const std::multimap<double, pair<Point2f, Point2f>>& distanceMap);
std::multimap<double, std::pair<cv::Point2f, cv::Point2f>> make_distance_map(const vector<Point2f>& srcPoints1, const vector<Point2f>& srcPoints2);
long double morph_distance(const std::vector<cv::Point2f>& srcPoints1, const std::vector<cv::Point2f>& srcPoints2, const long double& width, const long double& height, std::multimap<double, pair<Point2f, Point2f>> distanceMap = {});
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
void blur_margin(const Mat& src, const Size& szUnion, Mat& dst);
}
#endif
