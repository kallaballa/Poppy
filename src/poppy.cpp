#include "poppy.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>

#ifndef _WASM
#include <boost/program_options.hpp>
#endif

#include <opencv2/photo/photo.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#ifndef _WASM
namespace po = boost::program_options;
#endif

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
	srand(time(NULL));
#ifndef _WASM
	bool showGui = poppy::Settings::instance().show_gui;
	size_t numberOfFrames = poppy::Settings::instance().number_of_frames;
	double matchTolerance = poppy::Settings::instance().match_tolerance;
	double contourSensitivity = poppy::Settings::instance().contour_sensitivity;
	off_t maxKeypoints = poppy::Settings::instance().max_keypoints;
	bool autoTransform = poppy::Settings::instance().enable_auto_transform;
	bool srcScaling = false;
	bool denoise = false;
	string outputFile = "output.mkv";
	std::vector<string> imageFiles;
	po::options_description genericDesc("Options");
	genericDesc.add_options()
	("gui,g", "Show analysis windows.")
	("autotrans,a", "Try to automatically rotate and translate the source material to match.")
	("denoise,d", "Denoise images before morphing.")
	("scaling,s", "Instead of extending the source images, to match in size, use scaling.")
	("maxkey,m", po::value<off_t>(&maxKeypoints)->default_value(maxKeypoints), "Manual override for the number of keypoints to retain during detection. The default is automatic determination of that number.")
	("frames,f", po::value<size_t>(&numberOfFrames)->default_value(numberOfFrames), "The number of frames to generate.")
	("tolerance,t", po::value<double>(&matchTolerance)->default_value(matchTolerance), "How tolerant poppy is when matching keypoints.")
	("contour,c", po::value<double>(&contourSensitivity)->default_value(contourSensitivity), "How sensitive poppy is to contours.")
	("outfile,o", po::value<string>(&outputFile)->default_value(outputFile), "The name of the video file to write to.")
	("help,h", "Print the help message.");

	po::options_description hidden("Hidden options");
	hidden.add_options()("files", po::value<std::vector<string>>(&imageFiles), "image files");

	po::options_description cmdline_options;
	cmdline_options.add(genericDesc).add(hidden);

	po::positional_options_description p;
	p.add("files", -1);

	po::options_description visible;
	visible.add(genericDesc);

	po::variables_map vm;
	po::store(
			po::command_line_parser(argc, argv).options(cmdline_options).positional(
					p).run(), vm);
	po::notify(vm);

	if (vm.count("help") || imageFiles.empty()) {
		cerr << "Default options will work fine on good source material." << endl;
		cerr << "If you don't like the result you might try aligning the" << endl;
		cerr << "source images by hand. Anyway, there are also a couple" << endl;
		cerr << "of options you can specifiy. But usually you would only" << endl;
		cerr << "want to do this if you either have bad source material," << endl;
		cerr << "feel like experimenting or are trying to do something" << endl;
		cerr << "funny. The first thing to try is to adjust the match" << endl;
		cerr << "tolerance (--tolerance). If you still wanna tinker you" << endl;
		cerr << "should enable the gui (--gui) and play with the" << endl;
		cerr << "tolerance and maybe a little with contour sensitivity" << endl;
		cerr << "(--contour) and watch how it effects the algorithm." << endl;
		cerr << "Auto contour sensitivity works very well, that is why" << endl;
		cerr << "you probably shouldn't waste your time on it. One of" << endl;
		cerr << "goals of Poppy is to minimize manual parameters, so" << endl;
		cerr << "sooner or later parameter tinkering shouldn't be" << endl;
		cerr << "necessary anymore." << endl;
		cerr << "The key point limit (--maxkey) is useful for large" << endl;
		cerr << "images with lots of features which could easily yield" << endl;
		cerr << "two many keypoints for a particular machine. e.g. " << endl;
		cerr << "embedded systems. Please not that the cv::ORB" << endl;
		cerr << "extractor generates a larger number of key points than" << endl;
		cerr << "defined by this limit and only decides to retain that" << endl;
		cerr << "number in the end." << endl;
		cerr << visible;
		return 0;
	}
	if (vm.count("gui")) {
		showGui = true;
	}

	if (vm.count("autotrans")) {
		autoTransform = true;
	}

	if (vm.count("scaling")) {
		srcScaling = true;
	}

	if (vm.count("denoise")) {
		denoise = true;
	}

	for (auto p : imageFiles) {
		if (!std::filesystem::exists(p))
			throw std::runtime_error("File doesn't exist: " + p);
	}

	poppy::init(showGui, numberOfFrames, matchTolerance, contourSensitivity, maxKeypoints, autoTransform);
	Mat image1, denoise1;
	try {
		image1 = imread(imageFiles[0]);
		if(denoise) {
			fastNlMeansDenoising(image1, denoise1, 10,7,21);
			denoise1.copyTo(image1);
		}
		if (image1.empty()) {
			std::cerr << "Can't read (invalid?) image file: " + imageFiles[0] << std::endl;
			exit(2);
		}
	} catch (...) {
		std::cerr << "Can't read (invalid?) image file: " + imageFiles[0] << std::endl;
		exit(2);
	}

	Mat image2;

	Size szUnion = { image1.cols, image1.rows };
	for (size_t i = 1; i < imageFiles.size(); ++i) {
		Mat img = imread(imageFiles[i]);
		if(szUnion.width < img.cols) {
			szUnion.width = img.cols;
		}

		if(szUnion.height < img.rows) {
			szUnion.height = img.rows;
		}
	}

	Mat mUnion(szUnion.height, szUnion.width, image1.type(), {255,255,255});
	if(srcScaling) {
		Mat clone = image1.clone();
		resize(clone, image1, szUnion, INTER_CUBIC);
	} else {
		Rect centerRect((szUnion.width - image1.cols) / 2.0, (szUnion.height - image1.rows) / 2.0, image1.cols, image1.rows);
		image1.copyTo(mUnion(centerRect));
		image1 = mUnion.clone();
	}

	VideoWriter output(outputFile, VideoWriter::fourcc('F', 'F', 'V', '1'), 30,
			Size(szUnion.width, szUnion.height));

	for (size_t i = 1; i < imageFiles.size(); ++i) {
		Mat image2, denoise2;
		try {
			image2 = imread(imageFiles[i]);
			if(denoise) {
				std::cerr << "Denoising -> " << endl;
				fastNlMeansDenoising(image2, denoise2, 10,7,21);
				denoise2.copyTo(image2);
			}

			if (image2.empty()) {
				std::cerr << "Can't read (invalid?) image file: " + imageFiles[i] << std::endl;
				exit(2);
			}

			if(srcScaling) {
				Mat clone = image2.clone();
				resize(clone, image2, szUnion, INTER_CUBIC);
			} else {
				mUnion = Scalar::all(0);
				Rect cr((szUnion.width - image2.cols) / 2.0, (szUnion.height - image2.rows) / 2.0, image2.cols, image2.rows);
				image2.copyTo(mUnion(cr));
				image2 = mUnion.clone();
			}

			if (image1.cols != image2.cols || image1.rows != image2.rows) {
				std::cerr << "Image file sizes don't match: " << imageFiles[i] << std::endl;
				exit(3);
			}
		} catch (...) {
			std::cerr << "Can't read (invalid?) image file: " << imageFiles[i] << std::endl;
			exit(2);
		}
		std::cerr << "matching: " << imageFiles[i - 1] << " -> " << imageFiles[i] << " ..." << std::endl;
		poppy::morph(image1, image2, output);
		image1 = image2.clone();
	}
#endif
}
