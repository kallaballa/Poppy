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
	double phase = 1;
	double matchTolerance = poppy::Settings::instance().match_tolerance;
	double contourSensitivity = poppy::Settings::instance().contour_sensitivity;
	off_t maxKeypoints = poppy::Settings::instance().max_keypoints;
	bool autoAlign = poppy::Settings::instance().enable_auto_align;
	bool radial = poppy::Settings::instance().enable_radial_mask;
	bool face = poppy::Settings::instance().enable_face_detection;
	bool srcScaling = false;
	bool denoise = false;

	string outputFile = "output.mkv";
	std::vector<string> imageFiles;
	po::options_description genericDesc("Options");
	genericDesc.add_options()
	("gui,g", "Show analysis windows.")
	("face,e", "Enable face detection mode. Use if your source material consists of faces only.")
	("radial,r", "Use a radial mask to emphasize features in the center.")
	("autoalign,a", "Try to automatically align (rotate and translate) the source material to match.")
	("denoise,d", "Denoise images before morphing.")
	("scaling,s", "Instead of extending the source images, to match in size, use scaling.")
	("maxkey,m", po::value<off_t>(&maxKeypoints)->default_value(maxKeypoints), "Manual override for the number of keypoints to retain during detection. The default is automatic determination of that number.")
	("frames,f", po::value<size_t>(&numberOfFrames)->default_value(numberOfFrames), "The number of frames to generate.")
	("phase,p", po::value<double>(&phase)->default_value(phase), "A value from 0 to 1 telling poppy how far into the morph to start from.")
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
		cerr << "Usage: poppy [OPTIONS]... [IMAGEFILES]..." << endl;
		cerr << "Poppy automatically creates smooth transitions of shape" << endl;
		cerr << "and color between two images. That process is called " << endl;
		cerr << "image morphing and can be seen as a form of tweening or" << endl;
		cerr << "interpolation." << endl;
		cerr << endl;
		cerr << "Default options will work fine on good source material." << endl;
		cerr << "If you don't like the result you might try aligning the" << endl;
		cerr << "source images by hand (instead of using --autoalign). " << endl;
		cerr << "Anyway, there are also a couple of options you can" << endl;
		cerr << "specify. But usually you would only want to do this if" << endl;
		cerr << "you either have bad source material, feel like" << endl;
		cerr << "experimenting or are trying to do something funny." << endl;
		cerr << "The first thing to try is to adjust the match" << endl;
		cerr << "tolerance (--tolerance). If you want to tinker more," << endl;
		cerr << "You could enable the gui (--gui) and play with the" << endl;
		cerr << "tolerance and maybe a little with contour sensitivity" << endl;
		cerr << "(--contour) and watch how it effects the algorithm." << endl;
		cerr << "You probably shouldn't waste much time on the contour" << endl;
		cerr << "sensitivity parameter because it has little or even " << endl;
		cerr << "detrimental effect, which makes it virtually obsolete" << endl;
		cerr << "and it will be removed in the near future." << endl;
		cerr << "The key point limit (--maxkey) is useful for large" << endl;
		cerr << "images with lots of features which could easily yield" << endl;
		cerr << "two many keypoints for a particular machine. e.g. " << endl;
		cerr << "embedded systems. Please note that the feature extractor" << endl;
		cerr << "generates a larger number of key points than defined" << endl;
		cerr << "by this limit and only decides to retain that number" << endl;
		cerr << "in the end. The only means of reducing the number of" << endl;
		cerr << "generated key points at the moment is to denoise" << endl;
		cerr << "(--denoise) the source images. Obviously that is not" << endl;
		cerr << "optimal because you have no control over which" << endl;
		cerr << "features will be removed. Usually the parameter is used" << endl;
		cerr << "to enhance noisy images." << endl;
		cerr << visible;
		return 0;
	}
	if (vm.count("gui")) {
		showGui = true;
	}

	if (vm.count("autoalign")) {
		autoAlign = true;
	}

	if (vm.count("scaling")) {
		srcScaling = true;
	}

	if (vm.count("denoise")) {
		denoise = true;
	}

	if(vm.count("radial")) {
		radial = true;
	}

	if(vm.count("face")) {
		face = true;
	}

	for (auto p : imageFiles) {
		if (!std::filesystem::exists(p))
			throw std::runtime_error("File doesn't exist: " + p);
	}

	poppy::init(showGui, numberOfFrames, matchTolerance, contourSensitivity, maxKeypoints, autoAlign, radial, face);
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
		poppy::morph(image1, image2, phase, output);
		image1 = image2.clone();
	}
#endif
}
