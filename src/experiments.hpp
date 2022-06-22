#ifndef SRC_EXPERIMENTS_HPP_
#define SRC_EXPERIMENTS_HPP_

#include <opencv2/calib3d/calib3d.hpp>
#include "algo.hpp"
#include "util.hpp"

namespace poppy {
using namespace cv;
using namespace std;

int ratioTest(std::vector<std::vector<cv::DMatch> >
		&matches) {
#ifndef _NO_OPENCV
	int removed = 0;
	// for all matches
	for (std::vector<std::vector<cv::DMatch> >::iterator
	matchIterator = matches.begin();
			matchIterator != matches.end(); ++matchIterator) {
		// if 2 NN has been identified
		if (matchIterator->size() > 1) {
			// check distance ratio
			if ((*matchIterator)[0].distance /
					(*matchIterator)[1].distance > 0.7) {
				matchIterator->clear(); // remove match
				removed++;
			}
		} else { // does not have 2 neighbours
			matchIterator->clear(); // remove match
			removed++;
		}
	}
	return removed;
#else
        return 0;
#endif
}

cv::Mat ransacTest(
		const std::vector<cv::DMatch> &matches,
		const std::vector<cv::KeyPoint> &keypoints1,
		const std::vector<cv::KeyPoint> &keypoints2,
		std::vector<cv::DMatch> &outMatches)
		{
#ifndef _NO_OPENCV
	// Convert keypoints into Point2f
	std::vector<cv::Point2f> points1, points2;
	for (std::vector<cv::DMatch>::
	const_iterator it = matches.begin();
			it != matches.end(); ++it) {
		// Get the position of left keypoints
		float x = keypoints1[it->queryIdx].pt.x;
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x, y));
		// Get the position of right keypoints
		x = keypoints2[it->trainIdx].pt.x;
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x, y));
	}
	// Compute F matrix using RANSAC
	std::vector<uchar> inliers(points1.size(), 0);
	std::vector<cv::Point2f> out;
	//cv::Mat fundemental= cv::findFundamentalMat(points1, points2, out, CV_FM_RANSAC, 3, 0.99);

	cv::Mat fundemental = findFundamentalMat(
			cv::Mat(points1), cv::Mat(points2), // matching points
			inliers,      // match status (inlier or outlier)
			FM_RANSAC, // RANSAC method
			3.0,     // distance to epipolar line
			0.99);  // confidence probability

	// extract the surviving (inliers) matches
	std::vector<uchar>::const_iterator
	itIn = inliers.begin();
	std::vector<cv::DMatch>::const_iterator
	itM = matches.begin();
	// for all matches
	for (; itIn != inliers.end(); ++itIn, ++itM) {
		if (*itIn) { // it is a valid match
			outMatches.push_back(*itM);
		}
	}
	return fundemental;
#else
    return cv::Mat();
#endif
}

void symmetryTest(
		const std::vector<std::vector<cv::DMatch>> &matches1,
		const std::vector<std::vector<cv::DMatch>> &matches2,
		std::vector<cv::DMatch> &symMatches) {
	// for all matches image 1 -> image 2
	for (std::vector<std::vector<cv::DMatch>>::
	const_iterator matchIterator1 = matches1.begin();
			matchIterator1 != matches1.end(); ++matchIterator1) {
		// ignore deleted matches
		if (matchIterator1->size() < 2)
			continue;
		// for all matches image 2 -> image 1
		for (std::vector<std::vector<cv::DMatch>>::
		const_iterator matchIterator2 = matches2.begin();
				matchIterator2 != matches2.end();
				++matchIterator2) {
			// ignore deleted matches
			if (matchIterator2->size() < 2)
				continue;
			// Match symmetry test
			if ((*matchIterator1)[0].queryIdx ==
					(*matchIterator2)[0].trainIdx &&
					(*matchIterator2)[0].queryIdx ==
							(*matchIterator1)[0].trainIdx) {
				// add symmetrical match
				symMatches.push_back(
						cv::DMatch((*matchIterator1)[0].queryIdx,
								(*matchIterator1)[0].trainIdx,
								(*matchIterator1)[0].distance));
				break; // next match in image 1 -> image 2
			}
		}
	}
}


double morph_distance(const Mat &src1, const Mat &src2) {
	Mat tmp1, tmp2;
	if(src1.type() != CV_8UC1) {
		cvtColor(src1, tmp1, COLOR_RGB2GRAY);
	} else {
		tmp1 = src1;
	}

	if(src2.type() != CV_8UC1) {
		cvtColor(src2, tmp2, COLOR_RGB2GRAY);
	} else {
		tmp2 = src2;
	}

	vector<Point2f> corners1, corners2;
	goodFeaturesToTrack(tmp1, corners1, 25,0.01,10);
	goodFeaturesToTrack(tmp2, corners2, 25,0.01,10);

	if(corners1.empty() || corners2.empty())
		return numeric_limits<double>::max();

	if (corners1.size() > corners2.size())
		corners1.resize(corners2.size());
	else
		corners2.resize(corners1.size());


	match_points_by_proximity(corners1, corners2, src1.cols, src1.rows);

	if (corners1.size() > corners2.size())
		corners1.resize(corners2.size());
	else
		corners2.resize(corners1.size());

	return morph_distance(corners1, corners2, src1.cols, src1.rows);
}

void fft_shift(const Mat &input_img, Mat &output_img)
{
	output_img = input_img.clone();
	int cx = output_img.cols / 2;
	int cy = output_img.rows / 2;
	Mat q1(output_img, Rect(0, 0, cx, cy));
	Mat q2(output_img, Rect(cx, 0, cx, cy));
	Mat q3(output_img, Rect(0, cy, cx, cy));
	Mat q4(output_img, Rect(cx, cy, cx, cy));

	Mat temp;
	q1.copyTo(temp);
	q4.copyTo(q1);
	temp.copyTo(q4);
	q2.copyTo(temp);
	q3.copyTo(q2);
	temp.copyTo(q3);
}


void calculate_dft(Mat &src, Mat &dst) {
	Mat padded;
	int m = getOptimalDFTSize( src.rows );
	int n = getOptimalDFTSize( src.cols ); // on the border add zero values
	copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat z = Mat::zeros(m, n, CV_32F);
	Mat p;
	padded.convertTo(p, CV_32F);
	vector<Mat> planes = {p, z};
	Mat complexI;
	merge(planes, complexI);         // Add to the expanded another plane with zeros
	dft(complexI, complexI);         // this way the result may fit in the source matrix
	dst = complexI.clone();
}

Mat construct_H(Mat &scr, string type, float D0) {
	Mat H(scr.size(), CV_32F, Scalar(1));
	float D = 0;
	if (type == "Ideal") {
		for (int u = 0; u < H.rows; u++) {
			for (int  v = 0; v < H.cols; v++) {
				D = sqrt((u - scr.rows / 2)*(u - scr.rows / 2) + (v - scr.cols / 2)*(v - scr.cols / 2));
				if (D > D0)	{
					H.at<float>(u, v) = 0;
				}
			}
		}
		return (1.0 - H);
	}
	else if (type == "Gaussian") {
		for (int  u = 0; u < H.rows; u++) {
			for (int v = 0; v < H.cols; v++) {
				D = sqrt((u - scr.rows / 2)*(u - scr.rows / 2) + (v - scr.cols / 2)*(v - scr.cols / 2));
				H.at<float>(u, v) = exp(-D*D / (2 * D0*D0));
			}
		}
		return (1-0 - H);
	}
	assert(false);
	return H;
}


void fft_filter(Mat &scr, Mat &dst, Mat &H)
{
	fft_shift(H, H);
	Mat planesH[] = { Mat_<float>(H.clone()), Mat_<float>(H.clone()) };

	Mat planes_dft[] = { scr, Mat::zeros(scr.size(), CV_32F) };
	split(scr, planes_dft);

	Mat planes_out[] = { Mat::zeros(scr.size(), CV_32F), Mat::zeros(scr.size(), CV_32F) };
	planes_out[0] = planesH[0].mul(planes_dft[0]);
	planes_out[1] = planesH[1].mul(planes_dft[1]);

	merge(planes_out, 2, dst);

}
double dft_detail(const Mat& src, Mat& dst) {
	Mat imgIn;
	src.convertTo(imgIn, CV_32F);

	// DFT
	Mat DFT_image;
	calculate_dft(imgIn, DFT_image);

	// construct H
	Mat H;
	H = construct_H(imgIn, "Gaussian", 50);

	// filtering
	Mat complexIH;
	fft_filter(DFT_image, complexIH, H);

	// IDFT
	Mat imgOut;
	dft(complexIH, imgOut, DFT_INVERSE | DFT_REAL_OUTPUT);

//	normalize(imgOut, imgOut, 0, 1, NORM_MINMAX);
	dst = imgOut.clone();
	return countNonZero(1.0 - imgOut);
}

void correct_alignment(const Mat &src1, const Mat &src2, Mat &dst1, Mat &dst2, vector<vector<vector<Point2f>>> &collected1, vector<vector<vector<Point2f>>> &collected2) {
	cerr << "correcting alignment" << endl;
	assert(!collected1.empty() && !collected1[0].empty()
			&& !collected2.empty() && !collected2[0].empty());

	vector<Point2f> flat1;
	vector<Point2f> flat2;
	for (auto &c : collected1) {
		for (auto &v : c) {
			if (c.size() > 1) {
				RotatedRect rr = minAreaRect(v);
				if (rr.size.width > src1.cols * 0.5 || rr.size.height > src1.rows * 0.5) {
					continue;
				}
			}
			for (auto &pt : v) {
				flat1.push_back(pt);
			}
		}
	}

	if (flat1.empty())
		flat1 = collected1[0][0];

	for (auto &c : collected2) {
		for (auto &v : c) {
			if (c.size() > 1) {
				RotatedRect rr = minAreaRect(v);
				if (rr.size.width > src2.cols * 0.5 || rr.size.height > src2.rows * 0.5) {
					continue;
				}
			}
			for (auto &pt : v) {
				flat2.push_back(pt);
			}
		}
	}

	if (flat2.empty())
		flat2 = collected2[0][0];

	RotatedRect rr1 = minAreaRect(flat1);
	RotatedRect rr2 = minAreaRect(flat2);
	auto o1 = get_orientation(flat1);
	auto o2 = get_orientation(flat2);
	double angle1, angle2;
	o1.first = o1.first * 180 / M_PI;
	o2.first = o2.first * 180 / M_PI;
	o1.first = o1.first < 0 ? o1.first + 360 : o1.first;
	o2.first = o2.first < 0 ? o2.first + 360 : o2.first;

	if (fabs(o1.first - rr1.angle) < 22.5) {
		angle1 = (o1.first + rr1.angle) / 2.0;
	} else {
		double drr = fabs(rr1.angle - rr2.angle);
		double dor = fabs(o1.first - o2.first);
		if (dor < drr) {
			angle1 = dor;
		} else {
			angle1 = drr;
		}
	}

	if (fabs(o2.first - rr2.angle) < 22.5) {
		angle2 = (o2.first + rr2.angle) / 2.0;
	} else {
		double drr = fabs(rr1.angle - rr2.angle);
		double dor = fabs(o1.first - o2.first);
		if (dor < drr) {
			angle2 = dor;
		} else {
			angle2 = drr;
		}
	}

	double targetAng = angle2 - angle1;
	Point2f center1 = rr1.center;
	Point2f center2 = rr2.center;
	Mat rotated2;
	rotate(src2, rotated2, center2, targetAng);
	translate(rotated2, dst2, center1.x - center2.x, center1.y - center2.y);
	dst1 = src1.clone();

//	Mat test, copy1, copy2;
//	copy1 = dst1.clone();
//	copy2 = dst2.clone();
//	double lastDistance = numeric_limits<double>::max();
//	map<double, Mat> morphDistMap;
//	Mat distanceMap = Mat::zeros(copy1.rows, copy1.cols, CV_32FC1);
//	Mat preview;
//
//	for(int x = 0; x < copy1.cols; ++x) {
//		for(int y = 0; y < copy1.rows; ++y) {
//			translate(copy2, test,  x,  y);
//			distanceMap.at<float>(y,x) = cheap_morph_distance(copy1, test);
//			cerr << ((double(x * copy1.rows + y) / double(copy1.cols * copy1.rows)) * 100.0) << "%" << "\n";
//			normalize(distanceMap, preview, 0, 1, NORM_MINMAX);
//			show_image("DM",1.0 - preview);
//			waitKey(1);
//		}
//	}
//	cerr << endl;
//
//	while(true)
//		waitKey(0);


//	Mat t, test, grey1, grey2;
//	cvtColor(dst1, grey1, COLOR_RGB2GRAY);
//	cvtColor(dst2, grey2, COLOR_RGB2GRAY);
//
//	double d1, d2, d3, d4;
//	double lastDistance = numeric_limits<double>::max();
//	map<double, Mat> morphDistMap;
//
//	t = dst2.clone();
//	for(size_t i = 0; i < 100; ++i) {
//		morphDistMap.clear();
//		translate(t, test,  1,  0);
//		d1 = cheap_morph_distance(dst1, test);
//		morphDistMap[d1] = test;
//		translate(t, test, -1,  0);
//		d2 = cheap_morph_distance(dst1, test);
//		morphDistMap[d2] = test;
//		translate(t, test,  0,  1);
//		d3 = cheap_morph_distance(dst1, test);
//		morphDistMap[d3] = test;
//		translate(t, test,  0, -1);
//		d4 = cheap_morph_distance(dst1, test);
//		morphDistMap[d4] = test;
//		const auto& p = *morphDistMap.begin();
//		cerr << "searching: " << p.first << "\n";
//		if(p.first < lastDistance) {
//			cerr << "found: " << p.first << "\n";
//			lastDistance = p.first;
//			dst2 = p.second.clone();
//		}
//		t = p.second.clone();
//	}
//	cerr << endl;
}
}

#endif /* SRC_EXPERIMENTS_HPP_ */
