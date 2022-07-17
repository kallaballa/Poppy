#include "matcher.hpp"
#include "extractor.hpp"
#include "util.hpp"
#include "draw.hpp"
#include "settings.hpp"
#include "transformer.hpp"
#include "procrustes.hpp"
#include "terminal.hpp"

namespace poppy {

Matcher::~Matcher(){
}

void Matcher::find(Mat &corrected1, Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2) {
	Extractor extractor(img1_, img2_);
	extractor.prepareFeatures();
	Transformer trafo(img2_.cols, img2_.rows);

	if (ft1_.empty() || ft2_.empty()) {
		cerr << "general algorithm..." << endl;

		corrected1 = img1_.clone();
		corrected2 = img2_.clone();
		auto matches = extractor.keypointsRaw();
		srcPoints1 = matches.first;
		srcPoints2 = matches.second;
	} else {
		cerr << "face algorithm..." << endl;
		assert(!ft1_.empty() && !ft2_.empty());

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

	cerr << "keypoints remaining: " << srcPoints1.size() << "/" << srcPoints2.size() << endl;
	check_points(srcPoints1, img1_.cols, img1_.rows);
	check_points(srcPoints2, img1_.cols, img1_.rows);
}

void Matcher::autoAlign(Mat &corrected1, Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2) {
	double initialDist = morph_distance(srcPoints1, srcPoints2, img1_.cols, img1_.rows);
	cerr << "initial dist: " << initialDist << endl;
	double lastDistTrans = initialDist;
	double distTrans = initialDist;
	double lastDistRot = initialDist;
	double distRot = initialDist;
	double lastDistScale = initialDist;
	double distScale = initialDist;
	double lastDistProcr = initialDist;
	double distProcr = initialDist;

	Mat lastCorrected2;
	vector<Point2f> lastSrcPoints1, lastSrcPoints2;
	auto srcPointsRaw2 = srcPoints2;
	Transformer trafo(img1_.cols, img1_.rows);
	Terminal term;

	bool progress = false;
	do {
		progress = false;
		do {
			lastDistTrans = distTrans;
			lastCorrected2 = corrected2.clone();
			lastSrcPoints1 = srcPoints1;
			lastSrcPoints2 = srcPoints2;
			distTrans = trafo.retranslate(corrected2, srcPoints1, srcPoints2, srcPointsRaw2);
			if(distTrans < lastDistTrans) {
				progress = true;
				cerr << term.makeColor("retranslate dist: " + to_string(distTrans), Terminal::GREEN) << endl;
			} else {
				cerr << term.makeColor("retranslate dist: " + to_string(lastDistTrans), Terminal::RED) << endl;
			}
		} while(distTrans < lastDistTrans);

		if(distTrans >= lastDistTrans) {
			corrected2 = lastCorrected2.clone();
			srcPoints1 = lastSrcPoints1;
			srcPoints2 = lastSrcPoints2;
			distProcr = lastDistTrans;
		} else {
			distProcr = distTrans;
		}

		do {
			lastDistProcr = distProcr;
			lastCorrected2 = corrected2.clone();
			lastSrcPoints1 = srcPoints1;
			lastSrcPoints2 = srcPoints2;
			Procrustes procr(false, false);
			procr.procrustes(srcPoints1, srcPoints2);
			vector<Point2f> yPrime = procr.yPrimeAsVector();
			Mat perspectiveMat = getPerspectiveTransform(srcPoints2.data(), yPrime.data());
			perspectiveTransform(srcPoints2, srcPoints2, perspectiveMat);
			perspectiveMat.pop_back();
			warpAffine(corrected2, corrected2, perspectiveMat, corrected2.size());
			distProcr = morph_distance(srcPoints1, srcPoints2, img1_.cols, img1_.rows);
			if(distProcr < lastDistProcr) {
				progress = true;
				cerr << term.makeColor("procrustes dist:" + to_string(distProcr), Terminal::GREEN) << endl;
			} else {
				cerr << term.makeColor("procrustes dist:" + to_string(lastDistProcr), Terminal::RED) << endl;
			}
		} while(lastDistProcr > distProcr);

		if(distProcr >= lastDistProcr) {
			corrected2 = lastCorrected2.clone();
			srcPoints1 = lastSrcPoints1;
			srcPoints2 = lastSrcPoints2;
			distScale = lastDistProcr;
		} else {
			distScale = distProcr;
		}

		do {
			lastDistScale = distScale;
			lastCorrected2 = corrected2.clone();
			lastSrcPoints1 = srcPoints1;
			lastSrcPoints2 = srcPoints2;
			distScale = trafo.rescale(corrected2, srcPoints1, srcPoints2, srcPointsRaw2);
			if(distScale < lastDistScale) {
				progress = true;
				cerr << term.makeColor("rescale dist: " + to_string(distScale), Terminal::GREEN) << endl;
			} else {
				cerr << term.makeColor("rescale dist: " + to_string(lastDistScale), Terminal::RED) << endl;
			}
		} while(distScale < lastDistScale);

		if(distScale >= lastDistScale) {
			corrected2 = lastCorrected2.clone();
			srcPoints1 = lastSrcPoints1;
			srcPoints2 = lastSrcPoints2;
			distRot = lastDistScale;
		} else {
			distRot = distScale;
		}

		do {
			lastDistRot = distRot;
			lastCorrected2 = corrected2.clone();
			lastSrcPoints1 = srcPoints1;
			lastSrcPoints2 = srcPoints2;
			distRot = trafo.rerotate(corrected2, srcPoints1, srcPoints2, srcPointsRaw2);
			if(distRot < lastDistRot) {
				progress = true;
				cerr << term.makeColor("rerotate dist: " + to_string(distRot), Terminal::GREEN) << endl;
			} else {
				cerr << term.makeColor("rerotate dist: " + to_string(lastDistRot), Terminal::RED) << endl;
			}
		} while(distRot < lastDistRot);

		if(distRot >= lastDistRot) {
			corrected2 = lastCorrected2.clone();
			srcPoints1 = lastSrcPoints1;
			srcPoints2 = lastSrcPoints2;
			distScale = lastDistRot;
			distTrans = lastDistRot;
		} else {
			distScale = distRot;
			distTrans = distRot;
		}
	} while(progress);
}

void Matcher::match(Mat &corrected1, Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2) {
	assert(srcPoints1.size() == srcPoints2.size());
	assert(!srcPoints1.empty() && !srcPoints2.empty());

	multimap<double, pair<Point2f, Point2f>> distanceMap = make_distance_map(srcPoints1, srcPoints2);
	assert(!distanceMap.empty());

	auto distribution = calculate_sum_mean_and_sd(distanceMap);
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

	double thresh =
			((distanceMap.size() / density)
			* pow(mean / deviation,3)
			* (Settings::instance().match_tolerance)
			) / ((150 * (5 + sqrt(5)) / 2.0));

	srcPoints1.clear();
	srcPoints2.clear();
	double lastVal = 1.0;
	bool first = true;
	for (auto it = distanceMap.begin(); it != distanceMap.end(); ++it) {
		double value = (*it).first;
		double r = (value/thresh);

		if(first && r > 1) {
			first = false;
			thresh *= value / lastVal;
		}

		if(r > 0 && r <= 1.0) {
			srcPoints1.push_back((*it).second.first);
			srcPoints2.push_back((*it).second.second);
		}
		lastVal = value;
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

void Matcher::prepare(Mat &corrected1, Mat &corrected2, vector<Point2f> &srcPoints1, vector<Point2f> &srcPoints2) {
	//edit matches
	cerr << "matching: " << srcPoints1.size() << endl;
	match(corrected1, corrected2, srcPoints1, srcPoints2);
	cerr << "matched: " << srcPoints1.size() << endl;
	if (Settings::instance().enable_auto_align) {
		cerr << "auto aligning..." << endl;
		autoAlign(corrected1, corrected2, srcPoints1, srcPoints2);
	}

	Mat matMatches;
	Mat grey1, grey2;
	cvtColor(corrected1, grey1, COLOR_RGB2GRAY);
	cvtColor(corrected2, grey2, COLOR_RGB2GRAY);
	draw_matches(grey1, grey2, matMatches, srcPoints1, srcPoints2);
	show_image("x", matMatches);

	if (srcPoints1.size() > srcPoints2.size())
		srcPoints1.resize(srcPoints2.size());
	else
		srcPoints2.resize(srcPoints1.size());

	add_corners(srcPoints1, srcPoints2, corrected1.size);
}
} /* namespace poppy */

