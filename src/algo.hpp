#ifndef SRC_ALGO_HPP_
#define SRC_ALGO_HPP_
#include "blend.hpp"
#include "draw.hpp"
#include "util.hpp"

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


#ifndef _WASM
namespace po = boost::program_options;
#endif

typedef unsigned char sample_t;

using namespace std;
using namespace cv;
namespace poppy {
extern bool show_gui;
extern double number_of_frames;
extern size_t len_iterations;
extern double target_len_diff;
extern size_t ang_iterations;
extern double target_ang_diff;
extern double match_tolerance;
extern double contour_sensitivity;
extern off_t max_keypoints;
extern size_t pyramid_levels;

void canny_threshold(const Mat &src, Mat &detected_edges, double thresh);
void angle_test(std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols, int rows);
void angle_test(std::vector<Point2f> &ptv1, std::vector<Point2f> &ptv2, int cols, int rows);
void length_test(std::vector<std::tuple<KeyPoint, KeyPoint, double>> edges, std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols, int rows);
void length_test(std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols, int rows);
void length_test(std::vector<Point2f> &ptv1, std::vector<Point2f> &ptv2, int cols, int rows);
void make_delaunay_mesh(const Size &size, Subdiv2D &subdiv, std::vector<Point2f> &dstPoints);
std::pair<std::vector<Point2f>, std::vector<Point2f>> find_matches(const Mat &grey1, const Mat &grey2);
cv::Mat points_to_homogenous_mat(const std::vector<cv::Point2f> &pts);
void morph_points(std::vector<cv::Point2f> &srcPts1, std::vector<cv::Point2f> &srcPts2, std::vector<cv::Point2f> &dstPts, float s = 0.5);
void get_triangle_indices(const cv::Subdiv2D &subDiv, const std::vector<cv::Point2f> &points, std::vector<cv::Vec3i> &triangleVertices);
void make_triangler_points(const std::vector<cv::Vec3i> &triangleVertices, const std::vector<cv::Point2f> &points, std::vector<std::vector<cv::Point2f>> &trianglerPts);
void paint_triangles(cv::Mat &img, const std::vector<std::vector<cv::Point2f>> &triangles);
void solve_homography(const std::vector<cv::Point2f>& srcPts1, const std::vector<cv::Point2f>& srcPts2, cv::Mat& homography);
void solve_homography(const std::vector<std::vector<cv::Point2f>>& srcPts1,	const std::vector<std::vector<cv::Point2f>>& srcPts2, std::vector<cv::Mat>& hmats);
void morph_homography(const cv::Mat& hom, cv::Mat& morphHom1, cv::Mat& morphHom2, float blend_ratio);
void morph_homography(const std::vector<cv::Mat>& homs, std::vector<cv::Mat>& morphHoms1, std::vector<cv::Mat>& morphHoms2, float blend_ratio);
void create_map(const cv::Mat& triangleMap, const std::vector<cv::Mat>& homMatrices, cv::Mat& mapx, cv::Mat& mapy);
void draw_contour_map(std::vector<std::vector<std::vector<cv::Point>>>& collected, vector<Vec4i>& hierarchy, Mat &dst, int cols, int rows, int type);
void find_contours(const Mat& img1, const Mat& img2, std::vector<Mat>& dst1, std::vector<Mat>& dst2, Mat& allContours1, Mat& allContours2);
void find_matches(Mat &orig1, Mat &orig2, std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2, Mat &allContours1, Mat &allContours2);
std::tuple<double, double, double> calculate_sum_mean_and_sd(std::multimap<double, std::pair<Point2f, Point2f>> distanceMap);
void match_points_by_proximity(std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2, int cols, int rows);
void add_corners(std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2, MatSize sz);
void prepare_matches(Mat &origImg1, Mat &origImg2, const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2);
double morph_images(const Mat &origImg1, const Mat &origImg2, cv::Mat &dst, const cv::Mat &last, std::vector<cv::Point2f> &morphedPoints, std::vector<cv::Point2f> srcPoints1, std::vector<cv::Point2f> srcPoints2, Mat &allContours1, Mat &allContours2, double shapeRatio, double maskRatio);
}

#endif
