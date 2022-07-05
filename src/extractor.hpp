#ifndef SRC_EXTRACTOR_HPP_
#define SRC_EXTRACTOR_HPP_

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
	void foregroundMask(const Mat &grey, Mat &fgMask, const size_t& iterations = 12);
	void foreground(Mat &foreground1, Mat &foreground2);
public:
	Extractor(const Mat& img1, const Mat& img2);
	virtual ~Extractor();
	pair<vector<Point2f>, vector<Point2f>> keypointsRaw();
	pair<vector<Point2f>, vector<Point2f>> keypointsFlann();
	void contours(Mat &contourMap1, Mat &contourMap2, vector<Mat>& contourLayers1, vector<Mat>& contourLayers2);
};

} /* namespace poppy */

#endif /* SRC_EXTRACTOR_HPP_ */
