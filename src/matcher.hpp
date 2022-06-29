#ifndef SRC_MATCHER_HPP_
#define SRC_MATCHER_HPP_

#include <random>
#include <vector>
#include "face.hpp"
#include <opencv2/core/core.hpp>

using std::vector;
using std::pair;
using namespace cv;
namespace poppy {

class Matcher {
private:
	std::random_device rd;
	std::mt19937 g;
public:
	Matcher() : rd(), g(rd()) {
	}
	virtual ~Matcher();
	void find(const Mat &orig1, const Mat &orig2, Features& ft1, Features& ft2, Mat &corrected1, Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, Mat &contourMap1, Mat &contourMap2, Mat& plainContours1, Mat& plainContours2);
	void match(vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, int cols, int rows);
	void prepare(Mat &src1, Mat &src2, const Mat &img1, const Mat &img2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2);
};

} /* namespace poppy */

#endif /* SRC_MATCHER_HPP_ */
