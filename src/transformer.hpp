#ifndef SRC_TRANSFORMER_HPP_
#define SRC_TRANSFORMER_HPP_

#include "util.hpp"
#include "face.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace poppy {

class Transformer {
private:
	const size_t width_;
	const size_t height_;
public:
	Transformer(const size_t& width, const size_t& height);
	virtual ~Transformer();

	void translate(const Mat &src, Mat &dst, const Point2f& by);
	void rotate(const Mat &src, Mat &dst, Point2f center, double angle, double scale = 1);
	Point2f rotate_point(const cv::Point2f &inPoint, const double &angDeg);
	Point2f rotate_point(const cv::Point2f &inPoint, const cv::Point2f &center, const double &angDeg);
	void translate_points(vector<Point2f> &pts, const Point2f &by);
	void rotate_points(vector<Point2f> &pts, const Point2f &center, const double &angDeg);
	void scale_points(vector<Point2f> &pts, double coef);
	void translate_features(Features& ft, const Point2f &by);
	void scale_features(Features& ft, double coef);
	void rotate_features(Features& ft, const cv::Point2f &center, const double &angDeg);
	double retranslate(Mat &corrected2, vector<Point2f> &srcPointsFlann1, vector<Point2f> &srcPointsFlann2, vector<Point2f> &srcPointsRaw2);
	double rerotate(Mat &corrected2, vector<Point2f> &srcPointsFlann1, vector<Point2f> &srcPointsFlann2, vector<Point2f> &srcPointsRaw2);
};

} /* namespace poppy */

#endif /* SRC_TRANSFORMER_HPP_ */
