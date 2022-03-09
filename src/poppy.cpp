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

#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#ifndef _WASM
namespace po = boost::program_options;
#endif

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
	using std::string;
	srand(time(NULL));
	bool showGui = poppy::show_gui;
	size_t numberOfFrames = poppy::number_of_frames;
	double matchTolerance = poppy::match_tolerance;
	double targetAngDiff = poppy::target_ang_diff;
	double targetLenDiff = poppy::target_len_diff;
	double contourSensitivity = poppy::contour_sensitivity;
	off_t maxKeypoints = poppy::max_keypoints;
	std::vector<string> imageFiles;
	string outputFile = "output.mkv";
#ifndef _WASM
	po::options_description genericDesc("Options");
	genericDesc.add_options()
	("gui,g", "Show analysis windows")
	("maxkey,m", po::value<off_t>(&maxKeypoints)->default_value(maxKeypoints), "Manual override for the number of keypoints to retain during detection. The default is automatic determination of that number.")
	("frames,f", po::value<size_t>(&numberOfFrames)->default_value(numberOfFrames), "The number of frames to generate.")
	("tolerance,t", po::value<double>(&matchTolerance)->default_value(matchTolerance), "How tolerant poppy is when matching keypoints.")
	("angloss,a", po::value<double>(&targetAngDiff)->default_value(targetAngDiff), "The target loss, in percent, for the angle test.")
	("lenloss,l", po::value<double>(&targetLenDiff)->default_value(targetLenDiff), "The target loss, in percent, for the length test.")
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
		std::cerr << "Usage: poppy [options] <imageFiles>..." << std::endl;
		std::cerr << "Default options will work fine on good source material. Please," << std::endl;
		std::cerr << "always make sure images are scaled and rotated to match each" << std::endl;
		std::cerr << "other. Anyway, there are a couple of options you can specifiy." << std::endl;
		std::cerr << "But usually you would only want to do this if you either have" << std::endl;
		std::cerr << "bad source material, feel like experimenting or are trying to" << std::endl;
		std::cerr << "do something funny." << std::endl;
		std::cerr << visible;
		return 0;
	}

	if (vm.count("gui")) {
		showGui = true;
	}
#endif
	for (auto p : imageFiles) {
		if (!std::filesystem::exists(p))
			throw std::runtime_error("File doesn't exist: " + p);
	}

	poppy::init(showGui, numberOfFrames, matchTolerance, maxKeypoints, targetAngDiff, targetLenDiff, contourSensitivity);
	Mat image1;
	try {
		image1 = imread(imageFiles[0], cv::IMREAD_COLOR);
		if (image1.empty()) {
			std::cerr << "Can't read (invalid?) image file: " + imageFiles[0] << std::endl;
			exit(2);
		}
	} catch (...) {
		std::cerr << "Can't read (invalid?) image file: " + imageFiles[0] << std::endl;
		exit(2);
	}

	Mat image2;
	VideoWriter output(outputFile, VideoWriter::fourcc('F', 'F', 'V', '1'), 30,
			Size(image1.cols, image1.rows));

	for (size_t i = 1; i < imageFiles.size(); ++i) {
		Mat image2;
		try {
			image2 = imread(imageFiles[i], cv::IMREAD_COLOR);
			if (image2.empty()) {
				std::cerr << "Can't read (invalid?) image file: " + imageFiles[i] << std::endl;
				exit(2);
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
	}
}
