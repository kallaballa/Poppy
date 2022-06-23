#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

namespace poppy {
void draw_radial_gradiant2(Mat &grad);
void draw_radial_gradiant(Mat &grad);
void draw_delaunay(Mat &dst, const Size &size, Subdiv2D &subdiv, Scalar delaunay_color);
void draw_flow_heightmap(const Mat &morphed, const Mat &last, Mat &dst);
void draw_flow_vectors(const Mat &morphed, const Mat &last, Mat &dst);
void draw_flow_highlight(const Mat &morphed, const Mat &last, Mat &dst);
void draw_morph_analysis(const Mat &morphed, const Mat &last, Mat &dst, const Size &size, Subdiv2D &subdiv1, Subdiv2D &subdiv2, Subdiv2D &subdivMorph, Scalar delaunay_color);
void draw_matches(const Mat &grey1, const Mat &grey2, Mat &dst, std::vector<Point2f> &ptv1, std::vector<Point2f> &ptv2);
void draw_matches(const Mat &grey1, const Mat &grey2, Mat &dst, std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2);
void draw_optical_flow(const Mat &img1, const Mat &img2, Mat &dst);
}
