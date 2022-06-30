#include "matcher.hpp"
#include "extractor.hpp"
#include "util.hpp"
#include "draw.hpp"
#include "settings.hpp"
#include "transformer.hpp"

namespace poppy {

Matcher::~Matcher(){
}

void Matcher::find(const Mat &orig1, const Mat &orig2, Features& ft1, Features& ft2, Mat &corrected1, Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, Mat &contourMap1, Mat &contourMap2, Mat& edges1, Mat& edges2) {
	Extractor extractor;
	Transformer trafo;

	if (ft1.empty() || ft2.empty()) {
		cerr << "general algorithm..." << endl;
		Mat goodFeatures1, goodFeatures2;
		extractor.foreground(orig1, orig2, goodFeatures1, goodFeatures2);

		vector<Mat> contourLayers1;
		vector<Mat> contourLayers2;
		extractor.contours(orig1, orig2, contourMap1, contourMap2, edges1, edges2, contourLayers1, contourLayers2);

		corrected1 = orig1.clone();
		corrected2 = orig2.clone();

		show_image("gf1", goodFeatures1);
		show_image("gf2", goodFeatures2);

		if (Settings::instance().enable_auto_align) {
			cerr << "auto aligning..." << endl;

			srcPoints1.clear();
			srcPoints2.clear();
			auto matches = extractor.keypointsFlann(goodFeatures1, goodFeatures2);

			srcPoints1.insert(srcPoints1.end(), matches.first.begin(), matches.first.end());
			srcPoints2.insert(srcPoints2.end(), matches.second.begin(), matches.second.end());

			double lastDist = numeric_limits<double>::max();
			double dist = numeric_limits<double>::max();
			double globalDist = numeric_limits<double>::max();

			Mat lastCorrected2, lastContourMap2;
			vector<Point2f> lastSrcPoints1, lastSrcPoints2;

			do {
				do {
					lastDist = dist;
					lastCorrected2 = corrected2.clone();
					lastContourMap2 = contourMap2.clone();
					lastSrcPoints1 = srcPoints1;
					lastSrcPoints2 = srcPoints2;
					dist = trafo.retranslate(corrected2, contourMap2, srcPoints1, srcPoints2, contourMap1.cols, contourMap1.rows);
					cerr << "retranslate dist: " << dist << endl;
				} while(dist < lastDist);

				cerr << "final retranslate dist: " << lastDist << endl;
				corrected2 = lastCorrected2.clone();
				contourMap2  = lastContourMap2.clone();
				srcPoints1 = lastSrcPoints1;
				srcPoints2 = lastSrcPoints2;
				if(lastDist >= globalDist)
					break;

				globalDist = lastDist;

				lastDist = numeric_limits<double>::max();
				dist = numeric_limits<double>::max();
				do {
					lastDist = dist;
					lastCorrected2 = corrected2.clone();
					lastContourMap2 = contourMap2.clone();
					lastSrcPoints1 = srcPoints1;
					lastSrcPoints2 = srcPoints2;
					dist = trafo.rerotate(corrected2, contourMap2, srcPoints1, srcPoints2, contourMap1.cols, contourMap1.rows);
					cerr << "rerotate dist: " << dist << endl;
				} while(dist < lastDist);
				cerr << "final rerotate dist: " << lastDist << endl;
				corrected2 = lastCorrected2.clone();
				contourMap2  = lastContourMap2.clone();
				srcPoints1 = lastSrcPoints1;
				srcPoints2 = lastSrcPoints2;
				if(lastDist >= globalDist)
					break;
				globalDist = lastDist;
			} while(true);
			cerr << "final dist: " << globalDist << endl;

			Mat tf = getPerspectiveTransform(srcPoints1.data(), srcPoints2.data());
			srcPoints1.clear();
			srcPoints2.clear();
			matches = extractor.keypointsRaw(goodFeatures1, goodFeatures2);
			perspectiveTransform(matches.first, srcPoints1, tf);
			perspectiveTransform(matches.second, srcPoints2, tf);
		} else {
			srcPoints1.clear();
			srcPoints2.clear();

			auto matches = extractor.keypointsRaw(goodFeatures1, goodFeatures2);

			srcPoints1.insert(srcPoints1.end(), matches.first.begin(), matches.first.end());
			srcPoints2.insert(srcPoints2.end(), matches.second.begin(), matches.second.end());
		}
	} else {
		cerr << "face algorithm..." << endl;
		assert(!ft1.empty() && !ft2.empty());
		vector<Mat> contourLayers1;
		vector<Mat> contourLayers2;
		extractor.contours(orig1, orig2, contourMap1, contourMap2, edges1, edges2, contourLayers1, contourLayers2);

		if (Settings::instance().enable_auto_align) {
			cerr << "auto aligning..." << endl;

			double w1 = fabs(ft1.right_eye_[0].x - ft1.left_eye_[0].x);
			double w2 = fabs(ft2.right_eye_[0].x - ft2.left_eye_[0].x);
			double scale = w1 / w2;
			Mat scaledCorr2;
			Mat scaledCM2;

			resize(orig2, scaledCorr2, Size { int(std::round(orig2.cols * scale)), int(std::round(orig2.rows * scale)) });
			resize(contourMap2, scaledCM2, Size { int(std::round(orig2.cols * scale)), int(std::round(orig2.rows * scale)) });


			Point2f eyeVec1 = ft1.right_eye_[0] - ft1.left_eye_[0];
			Point2f eyeVec2 = ft2.right_eye_[0] - ft2.left_eye_[0];
			Point2f center1(ft1.left_eye_[0].x + (eyeVec1.x / 2.0), ft1.left_eye_[0].y + (eyeVec1.y / 2.0));
			Point2f center2(ft2.left_eye_[0].x + (eyeVec2.x / 2.0), ft2.left_eye_[0].y + (eyeVec2.y / 2.0));
			double angle1 = atan2(eyeVec1.y, eyeVec1.x);
			double angle2 = atan2(eyeVec2.y, eyeVec2.x);
			double dy = center1.y - center2.y;
			double dx = center1.x - center2.x;

			Mat translatedCorr2;
			Mat translatedCM2;
			trafo.translate(scaledCorr2, translatedCorr2, {float(dx), float(dy)});
			trafo.translate(scaledCM2, translatedCM2, {float(dx), float(dy)});

			angle1 = angle1 * 180 / M_PI;
			angle2 = angle2 * 180 / M_PI;
			angle1 = angle1 < 0 ? angle1 + 360 : angle1;
			angle2 = angle2 < 0 ? angle2 + 360 : angle2;
			double targetAng = angle2 - angle1;
			Mat rotatedCorr2;
			Mat rotatedCM2;

			trafo.rotate(translatedCorr2, rotatedCorr2, center2, targetAng);
			trafo.rotate(translatedCM2, rotatedCM2, center2, targetAng);

			corrected2 = orig2.clone();
			double dw = fabs(rotatedCorr2.cols - corrected2.cols);
			double dh = fabs(rotatedCorr2.rows - corrected2.rows);
			if(rotatedCorr2.cols > corrected2.cols) {
				rotatedCorr2(Rect(dw / 2, dh / 2, corrected2.cols, corrected2.rows)).copyTo(corrected2);
				rotatedCM2(Rect(dw / 2,  dh / 2, corrected2.cols, corrected2.rows)).copyTo(contourMap2);
			} else {
				rotatedCorr2.copyTo(corrected2(Rect(dw / 2, dh / 2, rotatedCorr2.cols, rotatedCorr2.rows)));
				rotatedCM2.copyTo(contourMap2(Rect(dw / 2,  dh / 2, rotatedCM2.cols, rotatedCM2.rows)));
			}
			corrected1 = orig1.clone();
			srcPoints1 = ft1.getAllPoints();
			ft2 = FaceDetector::instance().detect(corrected2);
			srcPoints2 = ft2.getAllPoints();
			assert(corrected1.cols == corrected2.cols && corrected1.rows == corrected2.rows);
		} else {
			srcPoints1 = ft1.getAllPoints();
			srcPoints2 = ft2.getAllPoints();

			corrected1 = orig1.clone();
			corrected2 = orig2.clone();
		}
	}
	filter_invalid_points(srcPoints1, srcPoints2, orig1.cols, orig1.rows);

	cerr << "keypoints: " << srcPoints1.size() << "/" << srcPoints2.size() << endl;
	check_points(srcPoints1, orig1.cols, orig1.rows);
	check_points(srcPoints2, orig1.cols, orig1.rows);
}

void Matcher::match(vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2, int cols, int rows) {
	multimap<double, pair<Point2f, Point2f>> distanceMap;
//	std::shuffle(srcPoints1.begin(), srcPoints1.end(), g);
//	std::shuffle(srcPoints2.begin(), srcPoints2.end(), g);

	Point2f nopoint(-1, -1);
	for (auto &pt1 : srcPoints1) {
		double dist = 0;
		double currentMinDist = numeric_limits<double>::max();

		Point2f *closest = &nopoint;
		for (auto &pt2 : srcPoints2) {
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
	assert(srcPoints1.size() == srcPoints2.size());
	assert(!srcPoints1.empty() && !srcPoints2.empty());

	auto distribution = calculate_sum_mean_and_sd(distanceMap);
	assert(!distanceMap.empty());
	srcPoints1.clear();
	srcPoints2.clear();

	double total = get<0>(distribution);
	double mean = get<1>(distribution);
	double deviation = get<2>(distribution);
	cerr << "size: " << distanceMap.size() << " total: " << total << " mean: " << mean << " deviation: " << deviation << endl;
	if(mean == 0) {
		for (auto it = distanceMap.rbegin(); it != distanceMap.rend(); ++it) {
			srcPoints1.push_back((*it).second.first);
			srcPoints2.push_back((*it).second.second);
		}
		assert(srcPoints1.size() == srcPoints2.size());
		assert(!srcPoints1.empty() && !srcPoints2.empty());

		return;
	}
	double value = 0;
	double limit = (mean / hypot(cols, rows)) * 1000;
	double limitCoef = 0.9;
	do {
		srcPoints1.clear();
		srcPoints2.clear();
		for (auto it = distanceMap.rbegin(); it != distanceMap.rend(); ++it) {
			value = (*it).first;

			if ((value > 0 && value < limit)) {
				srcPoints1.push_back((*it).second.first);
				srcPoints2.push_back((*it).second.second);
			}
		}
		cerr << "limit: " << limit << " coef: " << limitCoef << " points:" << srcPoints1.size() << " target: " << distanceMap.size() / (16.0 / (((deviation * hypot(cols, rows)) / total) * Settings::instance().match_tolerance)) << endl;
		assert(srcPoints1.size() == srcPoints2.size());
		check_points(srcPoints1, cols, rows);
		check_points(srcPoints2, cols, rows);
		if(srcPoints1.empty()) {
			limit *= (1.5 / limitCoef);
			limitCoef += ((1.0 - limitCoef) / 4.0);
			continue;
		} else {
			limit *= limitCoef;
		}

	} while (srcPoints1.empty() || srcPoints1.size() > (distanceMap.size() / (16.0 / (((deviation * hypot(cols, rows)) / total) * Settings::instance().match_tolerance))));

	assert(srcPoints1.size() == srcPoints2.size());
	assert(!srcPoints1.empty() && !srcPoints2.empty());
}

void Matcher::prepare(Mat &src1, Mat &src2, const Mat &img1, const Mat &img2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2) {
	//edit matches
	cerr << "matching: " << srcPoints1.size() << endl;
	match(srcPoints1, srcPoints2, img1.cols, img1.rows);
	cerr << "matched: " << srcPoints1.size() << endl;

	Mat matMatches;
	Mat grey1, grey2;
	cvtColor(src1, grey1, COLOR_RGB2GRAY);
	cvtColor(src2, grey2, COLOR_RGB2GRAY);
	draw_matches(grey1, grey2, matMatches, srcPoints1, srcPoints2);
	show_image("matched", matMatches);

	if (srcPoints1.size() > srcPoints2.size())
		srcPoints1.resize(srcPoints2.size());
	else
		srcPoints2.resize(srcPoints1.size());

	add_corners(srcPoints1, srcPoints2, src1.size);
}
} /* namespace poppy */
