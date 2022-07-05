#ifndef SRC_EXTRACTOR_HPP_
#define SRC_EXTRACTOR_HPP_

#include <vector>
#include <opencv2/core/core.hpp>

using std::vector;
using std::pair;
using namespace cv;

namespace poppy {

class Extractor {
public:
	Extractor();
	virtual ~Extractor();
	pair<vector<Point2f>, vector<Point2f>> keypointsRaw(const Mat &grey1, const Mat &grey2);
	pair<vector<Point2f>, vector<Point2f>> keypointsFlann(const Mat &grey1, const Mat &grey2);
	void foregroundMask(const Mat &grey, Mat &fgMask, const size_t& iterations = 12);
	void contours(const Mat &img1, const Mat &img2, Mat &contourMap1, Mat &contourMap2, vector<Mat>& contourLayers1, vector<Mat>& contourLayers2);
	void foreground(const Mat &img1, const Mat &img2, Mat &foreground1, Mat &foreground2);
};

} /* namespace poppy */

#endif /* SRC_EXTRACTOR_HPP_ */
