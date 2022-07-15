#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

namespace poppy {
void plot(Mat &img, vector<Point2f> points, Scalar color, int radius = 2);
void draw_radial_gradiant(Mat &grad);
Mat draw_radial_gradiant2(int width, int height);
void draw_contour_map(Mat &dst, vector<Mat>& contourLayers, const vector<vector<vector<Point2f>>> &collected, const vector<Vec4i> &hierarchy, int cols, int rows, int type);
void draw_delaunay(Mat &dst, const Size &size, Subdiv2D &subdiv, Scalar delaunay_color);
void draw_flow_heightmap(const Mat &morphed, const Mat &last, Mat &dst);
void draw_flow_vectors(const Mat &morphed, const Mat &last, Mat &dst);
void draw_flow_highlight(const Mat &morphed, const Mat &last, Mat &dst);
void draw_morph_analysis(const Mat &morphed, const Mat &last, Mat &dst, const Size &size, Subdiv2D &subdiv1, Subdiv2D &subdiv2, Subdiv2D &subdivMorph, Scalar delaunay_color);
void draw_matches(const Mat &grey1, const Mat &grey2, Mat &dst, std::vector<Point2f> &ptv1, std::vector<Point2f> &ptv2);
void draw_matches(const Mat &grey1, const Mat &grey2, Mat &dst, std::vector<KeyPoint> &kpv1, std::vector<KeyPoint> &kpv2);
void draw_optical_flow(const Mat &img1, const Mat &img2, Mat &dst);
}
