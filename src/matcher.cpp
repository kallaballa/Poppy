#include "matcher.hpp"
#include "extractor.hpp"
#include "util.hpp"
#include "draw.hpp"
#include "settings.hpp"
#include "transformer.hpp"

namespace poppy {

Matcher::~Matcher(){
}

void Matcher::find(Mat &corrected1, Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2) {
	Extractor extractor(img1_, img2_);
	extractor.prepareFeatures();
	Transformer trafo(img2_.cols, img2_.rows);

	if (ft1_.empty() || ft2_.empty()) {
		cerr << "general algorithm..." << endl;

//		extractor.contours(contourMap1, contourMap2, contourLayers1, contourLayers2);

		corrected1 = img1_.clone();
		corrected2 = img2_.clone();

		if (Settings::instance().enable_auto_align) {
			cerr << "auto aligning..." << endl;

			auto matchesFlann = extractor.keypointsFlann();
			auto matchesRaw = extractor.keypointsRaw();
			vector<Point2f> srcPointsRaw1 = matchesRaw.first;
			vector<Point2f> srcPointsRaw2 = matchesRaw.second;
			vector<Point2f> srcPointsFlann1 = matchesFlann.first;
			vector<Point2f> srcPointsFlann2 = matchesFlann.second;

			double lastDist = numeric_limits<double>::max();
			double dist = numeric_limits<double>::max();
			Mat lastCorrected2, lastContourMap2;
			vector<Point2f> lastSrcPoints1, lastSrcPoints2;
			cerr << "initial dist: " << morph_distance(srcPointsFlann1, srcPointsFlann2, img1_.cols, img1_.rows) << endl;
			do {
				lastDist = dist;
				lastCorrected2 = corrected2.clone();
				lastSrcPoints1 = srcPointsFlann1;
				lastSrcPoints2 = srcPointsFlann2;
				dist = trafo.retranslate(corrected2, srcPointsFlann1, srcPointsFlann2, srcPointsRaw2);
			} while(dist < lastDist);

			lastDist = dist;
			lastCorrected2 = corrected2.clone();
			lastSrcPoints1 = srcPointsFlann1;
			lastSrcPoints2 = srcPointsFlann2;
			dist = trafo.rerotate(corrected2, srcPointsFlann1, srcPointsFlann2, srcPointsRaw2);

			cerr << "retransform dist: " << dist << endl;

			if(dist > lastDist) {
				corrected2 = lastCorrected2.clone();
				srcPointsFlann1 = lastSrcPoints1;
				srcPointsFlann2 = lastSrcPoints2;
			}

			srcPoints1 = srcPointsRaw1;
			srcPoints2 = srcPointsRaw2;
		} else {
			auto matches = extractor.keypointsRaw();
			srcPoints1 = matches.first;
			srcPoints2 = matches.second;
		}
	} else {
		cerr << "face algorithm..." << endl;
		assert(!ft1_.empty() && !ft2_.empty());
//		extractor.contours(contourMap1, contourMap2, contourLayers1, contourLayers2);

		if (Settings::instance().enable_auto_align) {
			cerr << "auto aligning..." << endl;

			double w1 = fabs(ft1_.right_eye_[0].x - ft1_.left_eye_[0].x);
			double w2 = fabs(ft2_.right_eye_[0].x - ft2_.left_eye_[0].x);
			double scale = w1 / w2;
			Mat scaledCorr2;

			resize(img2_, scaledCorr2, Size { int(std::round(img2_.cols * scale)), int(std::round(img2_.rows * scale)) });

			Point2f eyeVec1 = ft1_.right_eye_[0] - ft1_.left_eye_[0];
			Point2f eyeVec2 = ft2_.right_eye_[0] - ft2_.left_eye_[0];
			Point2f center1(ft1_.left_eye_[0].x + (eyeVec1.x / 2.0), ft1_.left_eye_[0].y + (eyeVec1.y / 2.0));
			Point2f center2(ft2_.left_eye_[0].x + (eyeVec2.x / 2.0), ft2_.left_eye_[0].y + (eyeVec2.y / 2.0));
			double angle1 = atan2(eyeVec1.y, eyeVec1.x);
			double angle2 = atan2(eyeVec2.y, eyeVec2.x);
			double dy = center1.y - center2.y;
			double dx = center1.x - center2.x;

			Mat translatedCorr2;
			trafo.translate(scaledCorr2, translatedCorr2, {float(dx), float(dy)});

			angle1 = angle1 * 180 / M_PI;
			angle2 = angle2 * 180 / M_PI;
			angle1 = angle1 < 0 ? angle1 + 360 : angle1;
			angle2 = angle2 < 0 ? angle2 + 360 : angle2;
			double targetAng = angle2 - angle1;
			Mat rotatedCorr2;

			trafo.rotate(translatedCorr2, rotatedCorr2, center2, targetAng);

			corrected2 = img2_.clone();
			double dw = fabs(rotatedCorr2.cols - corrected2.cols);
			double dh = fabs(rotatedCorr2.rows - corrected2.rows);
			corrected2 = Scalar::all(0);
			if(rotatedCorr2.cols > corrected2.cols) {
				rotatedCorr2(Rect(dw / 2, dh / 2, corrected2.cols, corrected2.rows)).copyTo(corrected2);
			} else {
				rotatedCorr2.copyTo(corrected2(Rect(dw / 2, dh / 2, rotatedCorr2.cols, rotatedCorr2.rows)));
			}
			corrected1 = img1_.clone();
			srcPoints1 = ft1_.getAllPoints();
			ft2_ = FaceDetector::instance().detect(corrected2);
			srcPoints2 = ft2_.getAllPoints();
			assert(corrected1.cols == corrected2.cols && corrected1.rows == corrected2.rows);
		} else {
			cerr << "no alignment" << endl;
			srcPoints1 = ft1_.getAllPoints();
			srcPoints2 = ft2_.getAllPoints();

			corrected1 = img1_.clone();
			corrected2 = img2_.clone();
		}
	}
	filter_invalid_points(srcPoints1, srcPoints2, img1_.cols, img1_.rows);

	cerr << "keypoints: " << srcPoints1.size() << "/" << srcPoints2.size() << endl;
	check_points(srcPoints1, img1_.cols, img1_.rows);
	check_points(srcPoints2, img1_.cols, img1_.rows);
}

void Matcher::match(vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2) {
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
	double density = total/(img1_.cols * img1_.rows);

	cerr << "distance map size: " << distanceMap.size() << " density: " << density << " total: " << total << " mean: " << mean << " deviation: " << deviation << endl;
	if(mean == 0) {
		for (auto it = distanceMap.begin(); it != distanceMap.end(); ++it) {
			srcPoints1.push_back((*it).second.first);
			srcPoints2.push_back((*it).second.second);
		}

		assert(srcPoints1.size() == srcPoints2.size());
		assert(!srcPoints1.empty() && !srcPoints2.empty());

		return;
	}

	double thresh = 0.010 * (distanceMap.size() / density)
			* (mean / deviation)
		    * (Settings::instance().match_tolerance);

	srcPoints1.clear();
	srcPoints2.clear();

	auto it = distanceMap.begin();
	for (size_t i = 0; i < thresh; ++i) {
		srcPoints1.push_back((*it).second.first);
		srcPoints2.push_back((*it).second.second);
		advance(it, 1);
		if(it == distanceMap.end())
			break;
	}

	if(srcPoints1.empty()) {
		srcPoints1.push_back((*distanceMap.begin()).second.first);
		srcPoints2.push_back((*distanceMap.begin()).second.second);
	}
	assert(srcPoints1.size() == srcPoints2.size());
	check_points(srcPoints1, img1_.cols, img1_.rows);
	check_points(srcPoints2, img1_.cols, img1_.rows);

	assert(srcPoints1.size() == srcPoints2.size());
	assert(!srcPoints1.empty() && !srcPoints2.empty());
}

void Matcher::prepare(const Mat &corrected1, const Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2) {
	//edit matches
	cerr << "matching: " << srcPoints1.size() << endl;
	match(srcPoints1, srcPoints2);
	cerr << "matched: " << srcPoints1.size() << endl;

	Mat matMatches;
	Mat grey1, grey2;
	cvtColor(corrected1, grey1, COLOR_RGB2GRAY);
	cvtColor(corrected2, grey2, COLOR_RGB2GRAY);
	draw_matches(grey1, grey2, matMatches, srcPoints1, srcPoints2);
	show_image("matched", matMatches);

	if (srcPoints1.size() > srcPoints2.size())
		srcPoints1.resize(srcPoints2.size());
	else
		srcPoints2.resize(srcPoints1.size());

	add_corners(srcPoints1, srcPoints2, corrected1.size);
}
} /* namespace poppy */

