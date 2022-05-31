#ifndef SRC_ALGO_HPP_
#define SRC_ALGO_HPP_

#include <vector>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

namespace poppy {
void canny_threshold(const Mat &src, Mat &detected_edges, double thresh);
//void angle_test(std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols, int rows);
//void angle_test(std::vector<Point2f> &ptv1, std::vector<Point2f> &ptv2, int cols, int rows);
//void length_test(std::vector<std::tuple<KeyPoint, KeyPoint, double>> edges, std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols, int rows);
//void length_test(std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2, int cols, int rows);
//void length_test(std::vector<Point2f> &ptv1, std::vector<Point2f> &ptv2, int cols, int rows);
void make_delaunay_mesh(const Size &size, Subdiv2D &subdiv, std::vector<Point2f> &dstPoints);
std::pair<std::vector<Point2f>, std::vector<Point2f>> find_keypoints(const Mat &grey1, const Mat &grey2);
cv::Mat points_to_homogenous_mat(const std::vector<cv::Point2f> &pts);
void morph_points(std::vector<cv::Point2f> &srcPts1, std::vector<cv::Point2f> &srcPts2, std::vector<cv::Point2f> &dstPts, float s);
void get_triangle_indices(const cv::Subdiv2D &subDiv, const std::vector<cv::Point2f> &points, std::vector<cv::Vec3i> &triangleVertices);
void make_triangler_points(const std::vector<cv::Vec3i> &triangleVertices, const std::vector<cv::Point2f> &points, std::vector<std::vector<cv::Point2f>> &trianglerPts);
void paint_triangles(cv::Mat &img, const std::vector<std::vector<cv::Point2f>> &triangles);
void solve_homography(const std::vector<cv::Point2f>& srcPts1, const std::vector<cv::Point2f>& srcPts2, cv::Mat& homography);
void solve_homography(const std::vector<std::vector<cv::Point2f>>& srcPts1,	const std::vector<std::vector<cv::Point2f>>& srcPts2, std::vector<cv::Mat>& hmats);
void morph_homography(const cv::Mat& hom, cv::Mat& morphHom1, cv::Mat& morphHom2, float blend_ratio);
void morph_homography(const std::vector<cv::Mat>& homs, std::vector<cv::Mat>& morphHoms1, std::vector<cv::Mat>& morphHoms2, float blend_ratio);
void create_map(const cv::Mat& triangleMap, const std::vector<cv::Mat>& homMatrices, cv::Mat& mapx, cv::Mat& mapy);
void draw_contour_map(std::vector<std::vector<std::vector<cv::Point>>>& collected, vector<Vec4i>& hierarchy, Mat &dst, int cols, int rows, int type);
void extract_contours(const Mat &img1, const Mat &img2, Mat &allContours1, Mat &allContours2, vector<vector<vector<Point>>> &collected1, vector<vector<vector<Point>>> &collected2, vector<Mat> &contourLayers1, vector<Mat> &contourLayers2);
void extract_features(const Mat& img1, const Mat& img2, Mat &foreground1, Mat &foreground2, std::vector<Mat>& dst1, std::vector<Mat>& dst2, Mat& allContours1, Mat& allContours2, vector<vector<vector<Point>>>& collected1, vector<vector<vector<Point>>>& collected2, vector<Mat> &contourLayers1, vector<Mat> &contourLayers2);
void find_matches(const Mat &orig1, const Mat &orig2, Mat &corrected1, Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, Mat &allContours1, Mat &allContours2);
std::tuple<double, double, double> calculate_sum_mean_and_sd(std::multimap<double, std::pair<Point2f, Point2f>> distanceMap);
void match_points_by_proximity(std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2, int cols, int rows);
void add_corners(std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2, MatSize sz);
void prepare_matches(Mat &origImg1, Mat &origImg2, const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Point2f> &srcPoints1, std::vector<cv::Point2f> &srcPoints2);
double morph_images(const Mat &origImg1, const Mat &origImg2, cv::Mat &dst, const cv::Mat &last, std::vector<cv::Point2f> &morphedPoints, std::vector<cv::Point2f> srcPoints1, std::vector<cv::Point2f> srcPoints2, Mat &allContours1, Mat &allContours2, double shapeRatio, double maskRatio);
}

#endif
