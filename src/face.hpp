#ifndef SRC_FACE_HPP_
#define SRC_FACE_HPP_

#include <opencv2/dnn.hpp>
#include <vector>

namespace poppy {

using std::vector;
using cv::Point2f;

struct Features {
  vector<Point2f> chin;
  vector<Point2f> top_nose;
  vector<Point2f> bottom_nose;
  vector<Point2f> left_eyebrow;
  vector<Point2f> right_eyebrow;
  vector<Point2f> left_eye;
  vector<Point2f> right_eye;
  vector<Point2f> outer_lips;
  vector<Point2f> inside_lips;
};

class FaceDetector {
public:
    explicit FaceDetector();
    Features detect_face_rectangles(const cv::Mat &frame);
};

} /* namespace poppy */

#endif /* SRC_FACE_HPP_ */
