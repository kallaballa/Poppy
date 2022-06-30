#ifndef SRC_TRANSFORMER_HPP_
#define SRC_TRANSFORMER_HPP_

#include "util.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace poppy {

class Transformer {
public:
	Transformer();
	virtual ~Transformer();

	void translate(const Mat &src, Mat &dst, const Point2f& by);
	void rotate(const Mat &src, Mat &dst, Point2f center, double angle, double scale = 1);
	Point2f rotate_point(const cv::Point2f &inPoint, const double &angDeg);
	Point2f rotate_point(const cv::Point2f &inPoint, const cv::Point2f &center, const double &angDeg);
	void translate_points(vector<Point2f> &pts, const Point2f &by);
	void rotate_points(vector<Point2f> &pts, const Point2f &center, const double &angDeg);
	void scale_points(vector<Point2f> &pts, double coef);
	double retranslate(Mat &corrected2, Mat &contourMap2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, const size_t width, const size_t height);
	double rerotate(Mat &corrected2, Mat &contourMap2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, const size_t width, const size_t height);
};

} /* namespace poppy */

#endif /* SRC_TRANSFORMER_HPP_ */
