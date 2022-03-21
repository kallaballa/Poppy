#ifndef SRC_FACE_HPP_
#define SRC_FACE_HPP_

#include <opencv2/dnn.hpp>
#include <vector>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

namespace poppy {

using std::vector;
using cv::Point2f;
namespace d = dlib;

struct Features {
  vector<Point2f> chin_;
  vector<Point2f> top_nose_;
  vector<Point2f> bottom_nose_;
  vector<Point2f> left_eyebrow_;
  vector<Point2f> right_eyebrow_;
  vector<Point2f> left_eye_;
  vector<Point2f> right_eye_;
  vector<Point2f> outer_lips_;
  vector<Point2f> inside_lips_;

  vector<Point2f> getAllPoints() const {
	  vector<Point2f> allPoints;
	  allPoints.insert(allPoints.begin(), chin_.begin(), chin_.end());
	  allPoints.insert(allPoints.begin(), top_nose_.begin(), top_nose_.end());
	  allPoints.insert(allPoints.begin(), bottom_nose_.begin(), bottom_nose_.end());
	  allPoints.insert(allPoints.begin(), left_eyebrow_.begin(), left_eyebrow_.end());
	  allPoints.insert(allPoints.begin(), right_eyebrow_.begin(), right_eyebrow_.end());
	  allPoints.insert(allPoints.begin(), left_eye_.begin(), left_eye_.end());
	  allPoints.insert(allPoints.begin(), right_eye_.begin(), right_eye_.end());
	  allPoints.insert(allPoints.begin(), outer_lips_.begin(), outer_lips_.end());
	  allPoints.insert(allPoints.begin(), inside_lips_.begin(), inside_lips_.end());

	  return allPoints;
  }

  size_t empty() const {
	  return getAllPoints().empty();
  }
};

class FaceDetector {
public:
    explicit FaceDetector();
    Features detect(const cv::Mat &frame);
private:
	d::shape_predictor sp_;
	d::frontal_face_detector detector_;
};

} /* namespace poppy */

#endif /* SRC_FACE_HPP_ */
