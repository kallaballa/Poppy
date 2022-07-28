#include "util.hpp"
#include "settings.hpp"

#include <set>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

namespace poppy {

void color_reduce(Mat& image, int div=64) {
    int nl = image.rows;                    // number of lines
    int nc = image.cols * image.channels(); // number of elements per line

    for (int j = 0; j < nl; j++)
    {
        // get the address of row j
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++)
        {
            // process each pixel
            data[i] = data[i] / div * div + div / 2;
        }
    }
}

bool rect_contains(const Rect& r, const Point2f& pt, int radius) {
    return r.x - radius <= pt.x && pt.x  - radius < r.x + r.width && r.y - radius <= pt.y && pt.y - radius < r.y + r.height;
}

double euclidean_distance(cv::Point center, cv::Point point) {
	return hypot(center.x - point.x, center.y - point.y);
}

void gabor_filter(const Mat& src, Mat& dst, size_t numAngles, int kernel_size, double sig, double lm, double gm, double ps) {
    vector<double> theta(numAngles);

    float step = (180 / numAngles);
    for (size_t i = 0; i< numAngles; i++) {
        theta[i] = i * step;
    }

    dst = Mat::zeros(src.size(), src.type());

    for (size_t i = 0; i<numAngles; i++) {
        Mat plane;
        Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, theta[i], lm, gm, ps, CV_32F);
        filter2D(src, plane, CV_32F, kernel);
        plane.setTo(1.0, plane > 1.0);
        plane.setTo(0.0, plane < 0.0);
        dst += plane;
    }

    dst /= (numAngles);
}

void triple_channel(const Mat &src, Mat &dst) {
	vector<Mat> planes;
	for (int i = 0; i < 3; i++) {
		planes.push_back(src);
	}
	merge(planes, dst);
}

void equalize_bgr(const Mat &src, Mat &dst) {
	assert(src.type() == CV_8UC3);
	show_image("sr", dst);

	int histSize = 256;
	Mat grey;
	cvtColor(src, grey, COLOR_BGR2GRAY);

	float range[] = {0, 256};
	const float *histRange = {range};
	bool uniform = true;
	bool accumulate = false;

	cv::Mat hist;
	cv::calcHist(&grey, 1, 0, cv::Mat(), hist, 1, &histSize,
				 &histRange, uniform, accumulate);


	float max = 0;
	for(int i = 0; i < hist.rows; ++i) {
		max = std::max(max, hist.at<float>(i));
	}
	int count = 0;
	for(int i = 0; i < hist.rows; ++i) {
		if(hist.at<float>(i) > max / 2.0)
			++count;
	}

	dst = src.clone();
	color_reduce(dst, count);
	show_image("re", dst);

	Mat img_ycrcb;
	cvtColor(src, img_ycrcb, COLOR_BGR2YCrCb);
	vector<Mat> channels(3);
	split(img_ycrcb, channels);
	equalizeHist(channels[0], channels[0]);
	merge(channels, img_ycrcb);

	cvtColor(img_ycrcb, dst, COLOR_YCrCb2BGR);
	show_image("ds", dst);
}

Mat unsharp_mask(const Mat& original, float radius, float amount, float threshold) {
	assert(original.type() == CV_32FC3);

    cv::Mat input = original.clone();
    Mat retBuf = original.clone();

    // create the blurred copy
    Mat blurred;
    cv::GaussianBlur(input, blurred, cv::Size(0, 0), radius);

    // subtract blurred from original, pixel-by-pixel to make unsharp mask
    Mat unsharpMask;
    cv::subtract(input, blurred, unsharpMask);

    // --- filter on the mask ---

    cv::medianBlur(unsharpMask, unsharpMask, 3);
    //cv::blur(unsharpMask, unsharpMask, {3,3});

    // --- end filter ---

    // apply mask to image
    for (int row = 0; row < original.rows; row++)
    {
        for (int col = 0; col < original.cols; col++)
        {
            Vec3f origColor = input.at<Vec3f>(row, col);
            Vec3f difference = unsharpMask.at<Vec3f>(row, col);
            if(cv::norm(difference) >= threshold) {
            	retBuf.at<Vec3f>(row, col) = origColor + amount * difference;
            }
        }
    }

    return retBuf;
}

double feature_metric(const Mat &grey) {
	Mat corners;
	Mat unsharp = unsharp_mask(grey, 0.8, 12, 1.0);

	cornerHarris(unsharp, corners, 4, 9, 0.04);
	cv::Scalar mean, stddev;
	cv::meanStdDev(corners, mean, stddev);

	return stddev[0];
}

pair<double, Point2f> get_orientation(const vector<Point2f> &pts) {
	//Construct a buffer used by the pca analysis
	int sz = static_cast<int>(pts.size());
	Mat data_pts = Mat(sz, 2, CV_64F);
	for (int i = 0; i < data_pts.rows; i++)	{
		data_pts.at<double>(i, 0) = pts[i].x;
		data_pts.at<double>(i, 1) = pts[i].y;
	}
	//Perform PCA analysis
	PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);
	//Store the center of the object
	Point2f cntr = Point2f(pca_analysis.mean.at<double>(0, 0), pca_analysis.mean.at<double>(0, 1));
	//Store the eigenvalues and eigenvectors
	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; i++) {
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0), pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
	}

	return {atan2(eigen_vecs[0].y, eigen_vecs[0].x) * RADIANS_TO_DEGREES, cntr};
}

std::tuple<double, double, double> calculate_sum_mean_and_sd(multimap<double, pair<Point2f, Point2f>> distanceMap) {
	size_t s = distanceMap.size();
	double sum = 0.0, mean, standardDeviation = 0.0;

	for (auto &p : distanceMap) {
		sum += p.first;
	}

	mean = sum / s;

	for (auto &p : distanceMap) {
		standardDeviation += pow(p.first - mean, 2);
	}

	return {sum, mean, sqrt(standardDeviation / s)};
}

void add_corners(vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, MatSize sz) {
	float w = sz().width - 1;
	float h = sz().height - 1;
	srcPoints1.push_back(Point2f(0, 0));
	srcPoints1.push_back(Point2f(w, 0));
	srcPoints1.push_back(Point2f(0, h));
	srcPoints1.push_back(Point2f(w, h));
	srcPoints2.push_back(Point2f(0, 0));
	srcPoints2.push_back(Point2f(w, 0));
	srcPoints2.push_back(Point2f(0, h));
	srcPoints2.push_back(Point2f(w, h));
}

std::vector<Point2f> convertPointTo2f(const std::vector<Point>& pts) {
	std::vector<Point2f> result;
	for (auto &pt : pts) {
		result.push_back(Point2f(pt.x, pt.y));
	}
	return result;
}

std::vector<Point> convert2fToPoint(const std::vector<Point2f>& pts) {
	std::vector<Point> result;
	for (auto &pt : pts) {
		result.push_back(Point(pt.x, pt.y));
	}
	return result;
}

std::vector<std::vector<Point2f>> convertContourTo2f(const std::vector<std::vector<Point>> &contours1) {
	std::vector<std::vector<Point2f>> tmp;
	for (auto &vec : contours1) {
		std::vector<Point2f> row;
		for (auto &pt : vec) {
			row.push_back(Point2f(pt.x, pt.y));
		}
		if (!row.empty())
			tmp.push_back(row);
	}
	return tmp;
}

std::vector<std::vector<Point>> convertContourFrom2f(const std::vector<std::vector<Point2f>> &contours1) {
	std::vector<std::vector<Point>> tmp;
	for (auto &vec : contours1) {
		std::vector<Point> row;
		for (auto &pt : vec) {
			row.push_back(Point(pt.x, pt.y));
		}
		if (!row.empty())
			tmp.push_back(row);
	}
	return tmp;
}

void overdefineHull(vector<Point2f>& hull, size_t minPoints) {
	assert(hull.size() > 1);
	off_t diff = minPoints - hull.size();
	while(diff > 0)  {
		for(size_t i = 0; i < hull.size() && diff > 0; i+=2) {
			const auto& first = hull[i];
			const auto& second = hull[(i + 1) % hull.size()];
			auto insertee = first;
			auto vector = second - first;
			vector.x /= 2.0;
			vector.y /= 2.0;
			insertee.x += vector.x;
			insertee.y += vector.y;
			hull.insert(hull.begin() + (i+1), std::move(insertee));
			--diff;
		}
	}
}

pair<vector<Point2f>, vector<Point2f>> extract_points(const multimap<double, pair<Point2f, Point2f>>& distanceMap) {
	pair<vector<Point2f>, vector<Point2f>> res;
	for(const auto& p : distanceMap) {
		res.first.push_back(p.second.first);
		res.second.push_back(p.second.second);
	}
	return res;
}

multimap<double, pair<Point2f, Point2f>> make_distance_map(const vector<Point2f>& srcPoints1, const vector<Point2f>& srcPoints2) {
	multimap<double, pair<Point2f, Point2f>> distanceMap;
	auto copy1 = srcPoints1;
	auto copy2 = srcPoints2;

	Point2f nopoint(-1, -1);
	for (auto &pt1 : copy1) {
		double dist = 0;
		double currentMinDist = numeric_limits<double>::max();

		Point2f *closest = &nopoint;
		for (auto &pt2 : copy2) {
			if (pt2.x == -1 && pt2.y == -1)
				continue;

			dist = hypot(pt2.x - pt1.x, pt2.y - pt1.y);

			if (dist < currentMinDist) {
				currentMinDist = dist;
				closest = &pt2;
			}
		}
		if (closest->x == -1 && closest->y == -1)
			continue;

		dist = hypot(closest->x - pt1.x, closest->y - pt1.y);
		distanceMap.insert( { dist, { pt1, *closest } });
		closest->x = -1;
		closest->y = -1;
	}

	return distanceMap;
}

size_t increment = 1;

long double morph_distance(const vector<Point2f>& srcPoints1, const vector<Point2f>& srcPoints2, const long double& width, const long double& height, multimap<double, pair<Point2f, Point2f>> distanceMap) {
	assert(srcPoints1.size() == srcPoints2.size());
	if(distanceMap.empty())
		distanceMap = make_distance_map(srcPoints1, srcPoints2);

	vector<Point2f> hull1, hull2;
	vector<Point2f> contour1, contour2;
	convexHull(srcPoints1, hull1);
	convexHull(srcPoints2, hull2);
	approxPolyDP(Mat(hull1), contour1, 0.001, true);
	approxPolyDP(Mat(hull2), contour2, 0.001, true);
	auto area1 = fabs(contourArea(Mat(contour1)));
	auto area2 = fabs(contourArea(Mat(contour2)));

	float innerDist1 = 0;
	for(size_t i = 0; i < srcPoints1.size(); i+=increment) {
		const Point2f& p = srcPoints1[i];
		for(size_t j = 0; j < srcPoints1.size(); j+=increment) {
			const Point2f& v = p - srcPoints1[j];
			innerDist1 += v.x + v.y;
		}
	}
	innerDist1 = ((innerDist1 / (srcPoints1.size() * srcPoints1.size() * increment)) / (width + height));

	float innerDist2 = 0;
	for(size_t i = 0; i < srcPoints2.size(); i+=increment) {
		const Point2f& p = srcPoints2[i];
		for(size_t j = 0; j < srcPoints2.size(); j+=increment) {
			const Point2f& v = p - srcPoints1[j];
			innerDist2 += v.x + v.y;
		}
	}
	innerDist2 = ((innerDist2 / (srcPoints2.size() * srcPoints2.size() * increment)) / (width + height));

	float totalDistance = 0;
	for(const auto& p : distanceMap) {
		totalDistance += hypot(p.second.second.x - p.second.first.x, p.second.second.y - p.second.first.y);
	}

	auto ret = (((totalDistance / (distanceMap.size())) / hypot(width, height)) + fabs(innerDist1 - innerDist2) + (fabs(area1 - area2) / (width * height)) / 3.0);
	return ret;
}

long double morph_distance2(const vector<Point2f>& srcPoints1, const vector<Point2f>& srcPoints2, const long double& width, const long double& height, multimap<double, pair<Point2f, Point2f>> distanceMap) {
	assert(srcPoints1.size() == srcPoints2.size());
	if(distanceMap.empty())
		distanceMap = make_distance_map(srcPoints1, srcPoints2);

	float totalDistance = 0;
	for(const auto& p : distanceMap) {
		totalDistance += hypot(p.second.second.x - p.second.first.x, p.second.second.y - p.second.first.y);
	}
	auto ret = ((totalDistance / (distanceMap.size())) / hypot(width, height));
	assert(ret >= 0 && ret <= 1.0);
	return ret;
}

void show_image(const string &name, const Mat &img) {
#ifndef _WASM
	if(Settings::instance().show_gui) {
		namedWindow(name, WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
		imshow(name, img);
	}
#endif
}

void wait_key(int timeout) {
#ifndef _WASM
	if(Settings::instance().show_gui && Settings::instance().enable_wait) {
		int key = waitKey(timeout);
		while(key != (int) ('q')) {
			 key = waitKey(timeout);
		}
	}
#endif
}

void clip_points(std::vector<Point2f> &pts, int cols, int rows) {
	for (auto &pt : pts) {
		pt.x = pt.x > cols ? cols - 1 : pt.x;
		pt.y = pt.y > rows ? rows - 1 : pt.y;
		pt.x = pt.x < 0 ? 0 : pt.x;
		pt.y = pt.y < 0 ? 0 : pt.y;
	}
}

void check_points(const std::vector<Point2f> &pts, int cols, int rows) {
#ifndef NDEBUG
	for (const auto &pt : pts) {
		assert(!isinf(pt.x) && !isinf(pt.y));
		assert(!isnan(pt.x) && !isnan(pt.y));
		assert(pt.x >= 0 && pt.y >= 0);
		assert(pt.x <= cols && pt.y <= rows);
	}
#endif
}

void filter_invalid_points(vector<Point2f>& srcPoints1, vector<Point2f>& srcPoints2, int cols, int rows) {
	for (size_t i = 0; i < srcPoints1.size(); ++i) {
		Point2f& pt = srcPoints1[i];
		if(pt.x < 0 || pt.x > cols || pt.y < 0 || pt.y > rows) {
			srcPoints1.erase(srcPoints1.begin()+i);
			srcPoints2.erase(srcPoints2.begin()+i);
			--i;
		}
	}
	for (size_t i = 0; i < srcPoints2.size(); ++i) {
		Point2f& pt = srcPoints2[i];
		if(pt.x < 0 || pt.x > cols || pt.y < 0 || pt.y > rows) {
			srcPoints1.erase(srcPoints1.begin()+i);
			srcPoints2.erase(srcPoints2.begin()+i);
			--i;
		}
	}
}

void check_uniq(const std::vector<Point2f> &pts) {
#ifndef NDEBUG
	std::set<Point2f, LessPointOp> uniq;
	for (const auto &pt : pts) {
		assert(uniq.insert(pt).second);
	}
#endif
}

void check_min_distance(const std::vector<Point2f> &in, double minDistance) {
#ifndef NDEBUG
	double dist = 0;
	for (size_t i = 0; i < in.size(); ++i) {
		const auto& pt1 = in[i];
		for (size_t j = 0; j < in.size(); ++j) {
			if(i == j)
				continue;
			const auto& pt2 = in[j];
			dist = hypot(pt2.x - pt1.x, pt2.y - pt1.y);
			assert(dist >= minDistance);
		}
	}
#endif
}

void filter_min_distance(const std::vector<Point2f> &in, std::vector<Point2f> &out, double minDistance) {
	double dist = 0;
	for (size_t i = 0; i < in.size(); ++i) {
		const auto& pt1 = in[i];
		out.push_back(pt1);
		for (size_t j = i + 1; j < in.size(); ++j) {
			assert(i != j);
			const auto& pt2 = in[j];
			dist = hypot(pt2.x - pt1.x, pt2.y - pt1.y);
			if(dist >= minDistance) {
				out.push_back(pt2);
			}
		}
	}
	if(in.size() > out.size())
		cerr << "filtered: " << in.size() - out.size() << " points." << endl;
}

void make_uniq(const std::vector<Point2f> &pts, std::vector<Point2f> &out) {
	std::set<Point2f, LessPointOp> uniq;
	for (const auto &pt : pts) {
		if (uniq.insert(pt).second) {
			out.push_back(pt);
		}
	}
}

bool operator==(const Point2f &pt1, const Point2f &pt2) {
	return pt1.x == pt2.x && pt1.y == pt2.y;
}

bool operator==(const KeyPoint &kp1, const KeyPoint &kp2) {
	return kp1.pt.x == kp2.pt.x && kp1.pt.y == kp2.pt.y;
}

double distance(const Point2f &p1, const Point2f &p2) {
	return hypot(p2.x - p1.x, p2.y - p1.y);
}

cv::Point2f average(const std::vector<cv::Point2f> &pts) {
	Point2f result;
	for(const auto& pt : pts) {
		result += pt;
	}

	result.x /= pts.size();
	result.y /= pts.size();

	return result;
}

void blur_margin(const Mat& src, const Size& szUnion, Mat& dst) {
	Mat mUnion(szUnion.height, szUnion.width, src.type(), { 0, 0, 0 });
	int ksize = 127;
	int sigma = 6;
	double marginFactor = 1.3;
	double margin = (src.cols + src.rows) / 100.0;
	double dx = fabs(src.cols - szUnion.width) / 2.0;
	double dy = fabs(src.rows - szUnion.height) / 2.0;
	Rect roi(dx, dy, src.cols, src.rows);

	mUnion = Scalar::all(0);
	src.copyTo(mUnion(roi));

	dx = (dx == 0 ? marginFactor : dx + margin);
	dy = (dy == 0 ? marginFactor : dy + margin);
	Rect rl(0, 0, dx, szUnion.height);
	Rect rr(szUnion.width - dx, 0, dx, szUnion.height);
	Rect rt(0, 0, szUnion.width, dy);
	Rect rb(0, szUnion.height - dy, szUnion.width, dy);
	Mat left = mUnion(rl).clone();
	Mat right = mUnion(rr).clone();
	Mat top = mUnion(rt).clone();
	Mat bottom = mUnion(rb).clone();
	GaussianBlur(left, mUnion(rl), {ksize,ksize}, sigma);
	GaussianBlur(right, mUnion(rr), {ksize,ksize}, sigma);
	GaussianBlur(top, mUnion(rt), {ksize,ksize}, sigma);
	GaussianBlur(bottom, mUnion(rb), {ksize,ksize}, sigma);
	dst = mUnion.clone();
}
}
