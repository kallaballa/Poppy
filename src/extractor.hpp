#ifndef SRC_EXTRACTOR_HPP_
#define SRC_EXTRACTOR_HPP_

#include "settings.hpp"
#include <vector>
#include <opencv2/core/core.hpp>

using std::vector;
using std::pair;
using namespace cv;

namespace poppy {

class Extractor {
private:
	const Mat& img1_;
	const Mat& img2_;
	Mat grey1_;
	Mat grey2_;
	Mat goodFeatures1_;
	Mat goodFeatures2_;
	Mat contourMap1_;
	Mat contourMap2_;
public:
	Extractor(const Mat& img1, const Mat& img2);
	virtual ~Extractor();
	pair<Mat, Mat> prepareFeatures();

	pair<vector<KeyPoint>, vector<KeyPoint>> keypoints(const size_t& retainKeyPoints = 0);
	pair<vector<Point2f>, vector<Point2f>> points(const size_t& retainKeyPoints = 0);
	pair<vector<Point2f>, vector<Point2f>> eigen(const size_t& retainKeyPoints = 0);
	void foregroundMask(const Mat &grey, Mat &fgMask, const size_t& iterations = 12);
	void foreground(Mat &foreground1, Mat &foreground2);
};

} /* namespace poppy */

#endif /* SRC_EXTRACTOR_HPP_ */
