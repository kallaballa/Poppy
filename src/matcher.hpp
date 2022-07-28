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
	const Mat& img1_;
	const Mat& img2_;
	Features ft1_;
	Features ft2_;
	std::random_device rd;
	std::mt19937 g;
	double initialMorphDist_;
public:
	Matcher(const Mat& img1, const Mat& img2, const Features& ft1, const Features& ft2) :
		img1_(img1), img2_(img2), ft1_(ft1), ft2_(ft2), rd(), g(rd()), initialMorphDist_(0) {
	}
	virtual ~Matcher();
	void find(Mat &corrected1, Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2);
	void autoAlign(Mat &corrected1, Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2);
	void match(Mat &corrected1, Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2);
	void prepare(Mat &corrected1, Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2);
};

} /* namespace poppy */

#endif /* SRC_MATCHER_HPP_ */
