#ifndef SRC_ALGO_HPP_
#define SRC_ALGO_HPP_

#include <vector>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "face.hpp"

using namespace std;
using namespace cv;

namespace poppy {
void canny_threshold(const Mat &src, Mat &detected_edges, double thresh);
void make_delaunay_mesh(const Size &size, Subdiv2D &subdiv, std::vector<Point2f> &dstPoints);
cv::Mat points_to_homogenous_mat(const std::vector<cv::Point2f> &pts);
void morph_points(std::vector<cv::Point2f> &srcPts1, std::vector<cv::Point2f> &srcPts2, std::vector<cv::Point2f> &dstPts, float s);
void get_triangle_indices(const cv::Subdiv2D &subDiv, const std::vector<cv::Point2f> &points, std::vector<cv::Vec3i> &triangleVertices);
void make_triangler_points(const std::vector<cv::Vec3i> &triangleVertices, const std::vector<cv::Point2f> &points, std::vector<std::vector<cv::Point>> &trianglerPts);
void paint_triangles(cv::Mat &img, const std::vector<std::vector<cv::Point>> &triangles);
void solve_homography(const std::vector<cv::Point>& srcPts1, const std::vector<cv::Point>& srcPts2, cv::Mat& homography);
void solve_homography(const std::vector<std::vector<cv::Point>>& srcPts1,	const std::vector<std::vector<cv::Point>>& srcPts2, std::vector<cv::Mat>& hmats);
void morph_homography(const cv::Mat& hom, cv::Mat& morphHom1, cv::Mat& morphHom2, float blend_ratio);
void morph_homography(const std::vector<cv::Mat>& homs, std::vector<cv::Mat>& morphHoms1, std::vector<cv::Mat>& morphHoms2, float blend_ratio);
void create_map(const cv::Mat& triangleMap, const std::vector<cv::Mat>& homMatrices, cv::Mat& mapx, cv::Mat& mapy);
double morph_images(const Mat &origImg1, const Mat &origImg2, Mat &contourMap1, Mat &contourMap2, Mat &edges1, Mat &edges2, Mat &dst, const Mat &last, vector<Point2f> &morphedPoints, vector<Point2f> srcPoints1, vector<Point2f> srcPoints2, double shapeRatio, double maskRatio);
pair<double, Point2f> get_orientation(const vector<Point2f> &pts);
}

#endif
