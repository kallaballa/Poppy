#include "poppy.hpp"
#include "canvas.hpp"

#include <mutex>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <thread>
#include <algorithm>

#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include <SDL/SDL_stdinc.h>

#ifndef _WASM
#include <boost/program_options.hpp>
#else
#include <emscripten.h>
#endif
#include <gif.h>

GifWriter* gif_encoder = nullptr;

#include <opencv2/photo/photo.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#ifndef _WASM
namespace po = boost::program_options;
#endif

using namespace std;
using namespace cv;

volatile poppy::Canvas* canvas = nullptr;
volatile bool running = false;

std::mutex frameBufferMtx;
cv::Mat frameBuffer;

struct ChannelWriter {
	void write(Mat& mat) {
		std::unique_lock lock(frameBufferMtx);
		frameBuffer = mat.clone();
	}
};

struct SDLWriter {
	void write(Mat& mat) {
		if(mat.empty() || gif_encoder == nullptr || canvas == nullptr)
			return;
		Mat bgra;
		cvtColor(mat, bgra, COLOR_RGB2BGRA);
		if(gif_encoder != nullptr) {
			GifWriteFrame(gif_encoder, bgra.data, bgra.size().width, bgra.size().height, 100.0/poppy::Settings::instance().frame_rate);
		}
		Mat scaled;
		auto size = canvas->getSize();
		double w = size.first;
		double h = size.second;
		double sx = w / bgra.size().width;
		double sy = h / bgra.size().height;
		double factor = std::min(sx, sy);
		resize(bgra, scaled, Size(0, 0), factor, factor, INTER_LINEAR);
		if(factor < 1.0) {
			Mat background(h, w, CV_8UC4, Scalar(0,0,0,0));
			double borderX = (w - scaled.size().width) / 2.0;
			double borderY = (h - scaled.size().height) / 2.0;
			scaled.copyTo(background(Rect(borderX, borderY, scaled.size().width, scaled.size().height)));
			scaled = background.clone();
		}

		canvas->draw((image_t const&)scaled.data);
	}
};

SDLWriter sdl_writer;


void loop() {
#ifdef _WASM
	try {
		if(canvas != nullptr && running) {
			std::unique_lock lock(frameBufferMtx);
			sdl_writer.write(frameBuffer);
		}

//		if(!running) {
			SDL_Event event;
			SDL_PollEvent(&event);
			switch (event.type)
			{
			case SDL_WINDOWEVENT:
				if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
					canvas->resize(event.window.data1, event.window.data2);
				}
				break;
			default:
				break;
			}
//		}
	} catch (std::exception& ex) {
	std::cerr << "Main loop exception: " << ex.what() << std::endl;
	} catch (...) {
		std::cerr << "Main loop exception" << std::endl;
	}
#endif
}

Mat read_image(const string &path) {
	SDL_Surface *loadedSurface = IMG_Load(path.c_str());
	Mat result;
	if (loadedSurface == NULL) {
		printf("Unable to load image %s! SDL_image Error: %s\n", path.c_str(),
		IMG_GetError());
	} else {
		if (loadedSurface->w == 0 && loadedSurface->h == 0) {
			std::cerr << "Empty image loaded" << std::endl;
			SDL_FreeSurface(loadedSurface);
			return Mat();
		}
		if(loadedSurface->format->BytesPerPixel == 1) {
			result = Mat(loadedSurface->h, loadedSurface->w, CV_8UC1, (unsigned char*) loadedSurface->pixels, loadedSurface->pitch).clone();
			cvtColor(result,result, COLOR_GRAY2BGR);
		} else if(loadedSurface->format->BytesPerPixel == 3) {
			result = Mat(loadedSurface->h, loadedSurface->w, CV_8UC3, (unsigned char*) loadedSurface->pixels, loadedSurface->pitch).clone();
			if(loadedSurface->format->Rmask == 0x0000ff)
				cvtColor(result,result, COLOR_RGB2BGR);
		} else if(loadedSurface->format->BytesPerPixel == 4) {
			result = Mat(loadedSurface->h, loadedSurface->w, CV_8UC4, (unsigned char*) loadedSurface->pixels, loadedSurface->pitch).clone();
			if(loadedSurface->format->Rmask == 0x000000ff)
				cvtColor(result,result, COLOR_RGBA2BGR);
			else
				cvtColor(result,result, COLOR_RGBA2RGB);
		} else {
			std::cerr << "Unsupported image depth" << std::endl;
			SDL_FreeSurface(loadedSurface);
			return Mat();
		}
		SDL_FreeSurface(loadedSurface);
	}
	return result;
}

Size preserveAspect(const Size& origSize, const Size& extends) {
	double scale = std::min(extends.width / origSize.width, extends.height / origSize.height);
	return {int(origSize.width * scale), int(origSize.height * scale)};
}

void run(const std::vector<string> &imageFiles, const string &outputFile, double phase, bool distance) {
	cerr << "run" << endl;
	for (auto p : imageFiles) {
		if (!std::filesystem::exists(p))
			throw std::runtime_error("File doesn't exist: " + p);
	}

	Mat img1, denoise1;
	try {

		img1 = read_image(imageFiles[0]);
		if (poppy::Settings::instance().enable_denoise) {
			cerr << "denoising: " << imageFiles[0] << endl;
			fastNlMeansDenoising(img1, denoise1, 10, 7, 21);
			denoise1.copyTo(img1);
		}
		if (img1.empty()) {
			std::cerr << "Can't read (invalid?) image file: " + imageFiles[0] << std::endl;
			exit(2);
		}
	} catch (std::exception& ex) {
		std::cerr << "Can't read (invalid?) image file: " + imageFiles[0] << " Exception: " << ex.what() << std::endl;
		exit(2);
	}

	Mat img2;
	cerr << "building union" << endl;

	Size szUnion(0, 0);
	for (size_t i = 0; i < imageFiles.size(); ++i) {
		Mat img = read_image(imageFiles[i]);
		if (szUnion.width < img.cols) {
			szUnion.width = img.cols;
		}

		if (szUnion.height < img.rows) {
			szUnion.height = img.rows;
		}
		img.release();
	}
	cerr << "union: " << szUnion << endl;
	Mat mUnion(szUnion.height, szUnion.width, img1.type(), { 0, 0, 0 });
	if (poppy::Settings::instance().enable_src_scaling) {
		Mat clone = img1.clone();
		resize(clone, img1, preserveAspect(img1.size(), szUnion), INTER_LINEAR);
	}

	Rect centerRect((szUnion.width - img1.cols) / 2.0, (szUnion.height - img1.rows) / 2.0, img1.cols, img1.rows);
	img1.copyTo(mUnion(centerRect));
	img1 = mUnion.clone();

#ifndef _WASM
	string fourcc = poppy::Settings::instance().fourcc;
	if(fourcc.size() != 4)
		throw std::runtime_error("The fourcc identifier needs to be exactly four characters long.");

	VideoWriter output(outputFile, VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]), poppy::Settings::instance().frame_rate, Size(szUnion.width, szUnion.height));
#else

	{
	std::unique_lock lock(frameBufferMtx);
		if(gif_encoder != nullptr)
			delete gif_encoder;

		if(canvas != nullptr)
			delete canvas;

		canvas = new poppy::Canvas(350, 350, false);
		gif_encoder = new GifWriter();
		GifBegin(gif_encoder, "current.gif", szUnion.width, szUnion.height, 100.0/poppy::Settings::instance().frame_rate);
	}
	ChannelWriter output;
#endif
	cerr << "main loop..." << endl;

	for (size_t i = 1; i < imageFiles.size(); ++i) {
		Mat img2, denoise2;
		try {
			img2 = read_image(imageFiles[i]);
			if (poppy::Settings::instance().enable_denoise) {
				std::cerr << "Denoising -> " << endl;
				fastNlMeansDenoising(img2, denoise2, 10, 7, 21);
				denoise2.copyTo(img2);
			}

			if (0) {
				Mat grey1, grey2;
				Mat finalMask1Float, finalMask2Float;
				cvtColor(img1, grey1, COLOR_BGR2GRAY);
				cvtColor(img2, grey2, COLOR_BGR2GRAY);
				Mat fillMask1(grey1.rows + 2, grey1.cols + 2, grey1.type());
				Mat fillMask2(grey2.rows + 2, grey2.cols + 2, grey2.type());
				cv::Canny(grey1, fillMask1, 0, 255);
				cv::Canny(grey2, fillMask2, 0, 255);
				cv::copyMakeBorder(fillMask1, fillMask1, 1, 1, 1, 1, cv::BORDER_REPLICATE);
				cv::copyMakeBorder(fillMask2, fillMask2, 1, 1, 1, 1, cv::BORDER_REPLICATE);
				uchar fillValue = 127;
				cv::floodFill(grey1, fillMask1, { 1, 1 }, cv::Scalar(127), 0, cv::Scalar(), cv::Scalar(), 4 | cv::FLOODFILL_MASK_ONLY | (fillValue << 8));
				cv::floodFill(grey1, fillMask1, { 1, grey1.rows - 1 }, cv::Scalar(127), 0, cv::Scalar(), cv::Scalar(), 4 | cv::FLOODFILL_MASK_ONLY | (fillValue << 8));
				cv::floodFill(grey1, fillMask1, { grey1.cols - 1, 1 }, cv::Scalar(127), 0, cv::Scalar(), cv::Scalar(), 4 | cv::FLOODFILL_MASK_ONLY | (fillValue << 8));
				cv::floodFill(grey1, fillMask1, { grey1.cols - 1, grey1.rows - 1 }, cv::Scalar(127), 0, cv::Scalar(), cv::Scalar(), 4 | cv::FLOODFILL_MASK_ONLY | (fillValue << 8));

				cv::floodFill(grey2, fillMask2, { 1, 1 }, cv::Scalar(127), 0, cv::Scalar(), cv::Scalar(), 4 | cv::FLOODFILL_MASK_ONLY | (fillValue << 8));
				cv::floodFill(grey2, fillMask2, { 1, grey2.rows - 1 }, cv::Scalar(127), 0, cv::Scalar(), cv::Scalar(), 4 | cv::FLOODFILL_MASK_ONLY | (fillValue << 8));
				cv::floodFill(grey2, fillMask2, { grey2.cols - 1, 1 }, cv::Scalar(127), 0, cv::Scalar(), cv::Scalar(), 4 | cv::FLOODFILL_MASK_ONLY | (fillValue << 8));
				cv::floodFill(grey2, fillMask2, { grey2.cols - 1, grey2.rows - 1 }, cv::Scalar(127), 0, cv::Scalar(), cv::Scalar(), 4 | cv::FLOODFILL_MASK_ONLY | (fillValue << 8));

				Mat fillThresh1, fillThresh2;
				inRange(fillMask1, { 126 }, { 128 }, fillThresh1);
				inRange(fillMask2, { 126 }, { 128 }, fillThresh2);

				Mat test1, test2;
				threshold(fillThresh1, test1, 0, 1, THRESH_BINARY);
				threshold(fillThresh2, test2, 0, 1, THRESH_BINARY);
				fillThresh1 = 255.0 - fillThresh1;
				fillThresh2 = 255.0 - fillThresh2;

				if (countNonZero(test1) > (test1.cols * test1.rows * 0.2)) {
					fillThresh1(Rect(1, 1, img1.cols, img1.rows)).convertTo(finalMask1Float, CV_32F, 1.0 / 255.0);
				} else {
					Mat hue;
					int bins = 6;
					Mat hsv;
					cvtColor(img1, hsv, COLOR_BGR2HSV);
					hue.create(hsv.size(), hsv.depth());
					int ch[] = { 0, 0 };
					mixChannels(&hsv, 1, &hue, 1, ch, 1);
					int histSize = MAX(bins, 2);
					float hue_range[] = { 0, 180 };
					const float *ranges[] = { hue_range };
					Mat hist;
					calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);
					normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
					Mat backproj;
					calcBackProject(&hue, 1, 0, hist, backproj, ranges, 1, true);
					Mat filtered;
					Mat filteredFloat;
					inRange(backproj, { 254 }, { 255 }, filtered);
					filtered.convertTo(finalMask1Float, CV_32F, 1 / 255.0);
				}

				if (countNonZero(test2) > (test2.cols * test2.rows * 0.2)) {
					fillThresh2(Rect(1, 1, img2.cols, img2.rows)).convertTo(finalMask2Float, CV_32F, 1.0 / 255.0);
				} else {
					Mat hue;
					int bins = 6;
					Mat hsv;
					cvtColor(img2, hsv, COLOR_BGR2HSV);
					hue.create(hsv.size(), hsv.depth());
					int ch[] = { 0, 0 };
					mixChannels(&hsv, 1, &hue, 1, ch, 1);
					int histSize = MAX(bins, 2);
					float hue_range[] = { 0, 180 };
					const float *ranges[] = { hue_range };
					Mat hist;
					calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);
					normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
					Mat backproj;
					calcBackProject(&hue, 1, 0, hist, backproj, ranges, 1, true);
					Mat filtered;
					Mat filteredFloat;
					inRange(backproj, { 254 }, { 255 }, filtered);
					filtered.convertTo(finalMask2Float, CV_32F, 1 / 255.0);
				}
				Mat img1Float, img2Float;
				img1.convertTo(img1Float, CV_32F, 1 / 255.0);
				img2.convertTo(img2Float, CV_32F, 1 / 255.0);

				Mat finalMask1FloatC3 = Mat::zeros(grey1.rows, grey1.cols, CV_32FC3);
				Mat finalMask2FloatC3 = Mat::zeros(grey2.rows, grey2.cols, CV_32FC3);
				Mat dilated1, dilated2;
				dilate(finalMask1Float, dilated1, getStructuringElement(MORPH_ELLIPSE, Size(11, 11)));
				dilate(finalMask2Float, dilated2, getStructuringElement(MORPH_ELLIPSE, Size(11, 11)));
				GaussianBlur(dilated1, finalMask1Float, { 127, 127 }, 12);
				GaussianBlur(dilated2, finalMask2Float, { 127, 127 }, 12);

				vector<Mat> planes1, planes2;
				for (int i = 0; i < 3; i++) {
					planes1.push_back(finalMask1Float);
					planes2.push_back(finalMask2Float);
				}
				merge(planes1, finalMask1FloatC3);
				merge(planes2, finalMask2FloatC3);

				Mat blurred1Float, blurred2Float;
				Mat maskedBlur1Float, maskedBlur2Float;
				Mat masked1Float, masked2Float;
				Mat invFinalMask1 = Scalar(1.0, 1.0, 1.0) - finalMask1FloatC3;
				Mat invFinalMask2 = Scalar(1.0, 1.0, 1.0) - finalMask2FloatC3;
				Mat combined1, combined2;
				GaussianBlur(img1Float, blurred1Float, { 63, 63 }, 5);
				GaussianBlur(img2Float, blurred2Float, { 63, 63 }, 5);
				multiply(blurred1Float, invFinalMask1, maskedBlur1Float);
				multiply(blurred2Float, invFinalMask2, maskedBlur2Float);
				multiply(img1Float, finalMask1FloatC3, masked1Float);
				multiply(img2Float, finalMask2FloatC3, masked2Float);
				add(masked1Float, maskedBlur1Float, combined1);
				add(masked2Float, maskedBlur2Float, combined2);

//			poppy::show_image("fm1", finalMask1FloatC3);
//			poppy::show_image("fm2", finalMask2FloatC3);
//			poppy::show_image("iv1", invFinalMask1);
//			poppy::show_image("iv2", invFinalMask2);
//			poppy::show_image("b1", blurred1Float);
//			poppy::show_image("b2", blurred2Float);
//			poppy::show_image("mb1", maskedBlur1Float);
//			poppy::show_image("mb2", maskedBlur2Float);
//			poppy::show_image("ig1", img1Float);
//			poppy::show_image("ig2", img2Float);
//			poppy::show_image("mf1", masked1Float);
//			poppy::show_image("mf2", masked2Float);
//			poppy::show_image("c1", combined1);
//			poppy::show_image("c2", combined2);
//			poppy::wait_key();
				combined1.convertTo(img1, CV_8UC3, 255);
				combined2.convertTo(img2, CV_8UC3, 255);
			}
			if (img2.empty()) {
				std::cerr << "Can't read (invalid?) image file: " + imageFiles[i] << std::endl;
				exit(2);
			}

			if (poppy::Settings::instance().enable_src_scaling) {
				Mat bg = Mat::zeros(szUnion, img2.type());
				Mat clone = img2.clone();
				Size aspect = preserveAspect(img2.size(), szUnion);
				resize(clone, img2, aspect, INTER_LINEAR);
				double dx = fabs(aspect.width - szUnion.width);
				double dy = fabs(aspect.height - szUnion.height);
				img2.copyTo(bg(Rect(dx / 2.0, dy / 2.0, aspect.width, aspect.height)));
				img2 = bg.clone();
				bg.release();
				clone.release();
			} else {
				mUnion = Scalar::all(0);
				Rect cr((szUnion.width - img2.cols) / 2.0, (szUnion.height - img2.rows) / 2.0, img2.cols, img2.rows);
				img2.copyTo(mUnion(cr));
				img2 = mUnion.clone();
			}

			if (img1.cols != img2.cols || img1.rows != img2.rows) {
				std::cerr << "Image file sizes don't match: " << img1.size() << "/" << img2.size() << "/" << szUnion << std::endl;
				exit(3);
			}
		} catch (std::exception& ex) {
			std::cerr << "Can't read (invalid?) image file: " + imageFiles[0] << " Exception: " << ex.what() << std::endl;
			exit(2);
		}
		std::cerr << "matching: " << imageFiles[i - 1] << " -> " << imageFiles[i] << " ..." << std::endl;

		poppy::morph(img1, img2, phase, distance, output);
		img1 = img2.clone();
		img2.release();
	}
#ifdef _WASM
	cerr << "gif flush" << endl;
	try {
		{
			std::unique_lock lock(frameBufferMtx);
			cerr << "gif result:" << GifEnd(gif_encoder) << endl;
			frameBuffer = Mat();
		}
	} catch(...) {
		cerr << "gif end failed" << endl;
	}
#endif
	cerr << "done" << endl << flush;
}

int main(int argc, char **argv) {
	srand(time(NULL));

#ifndef _WASM
	bool showGui = poppy::Settings::instance().show_gui;
	size_t numberOfFrames = poppy::Settings::instance().number_of_frames;
	size_t pyramidLevels = poppy::Settings::instance().pyramid_levels;
	double frameRate = poppy::Settings::instance().frame_rate;
	double phase = -1;
	double matchTolerance = poppy::Settings::instance().match_tolerance;
	double contourSensitivity = poppy::Settings::instance().contour_sensitivity;
	off_t maxKeypoints = poppy::Settings::instance().max_keypoints;
	bool autoAlign = poppy::Settings::instance().enable_auto_align;
	bool radial = poppy::Settings::instance().enable_radial_mask;
	bool face = poppy::Settings::instance().enable_face_detection;
	bool srcScaling = true;
	bool denoise = poppy::Settings::instance().enable_denoise;
	bool distance = false;
	string fourcc = poppy::Settings::instance().fourcc;
	std::vector<string> imageFiles;
	string outputFile = "output.mkv";

	po::options_description genericDesc("Options");
	genericDesc.add_options()
	("gui,g", "Show analysis windows.")
	("face,e", "Enable face detection mode. Use if your source material consists of faces only.")
	("radial,r", "Use a radial mask to emphasize features in the center.")
	("autoalign,a", "Try to automatically align (rotate and translate) the source material to match.")
	("denoise,d", "Denoise images before morphing.")
	("distance,n", "Calculate the morph distance and return.")
	("scaling,s", "Instead of extending the source images, to match in size, use scaling.")
	("maxkey,m", po::value<off_t>(&maxKeypoints)->default_value(maxKeypoints), "Manual override for the number of keypoints to retain during detection. The default is automatic determination of that number.")
	("rate,b", po::value<double>(&frameRate)->default_value(frameRate), "The frame rate of the output video.")
	("frames,f", po::value<size_t>(&numberOfFrames)->default_value(numberOfFrames), "The number of frames to generate.")
	("phase,p", po::value<double>(&phase)->default_value(phase), "A value from 0 to 1 telling poppy how far into the morph to start from.")
	("pyramid,y", po::value<size_t>(&pyramidLevels)->default_value(pyramidLevels), "How many levels to use for the laplacian pyramid.")
	("tolerance,t", po::value<double>(&matchTolerance)->default_value(matchTolerance), "How tolerant poppy is when matching keypoints.")
	("contour,c", po::value<double>(&contourSensitivity)->default_value(contourSensitivity), "How sensitive poppy is to contours.")
	("fourcc,u", po::value<string>(&fourcc)->default_value(fourcc), "The four letter fourcc identifier (https://en.wikipedia.org/wiki/FourCC) which selects the video format. e.g: \"FFV1\", \"h264\", \"theo\"")
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
		cerr << "Anyway, you probably shouldn't waste much time on the" << endl;
		cerr << "contour sensitivity parameter because it has little or" << endl;
		cerr << "even detrimental effect, which makes it virtually" << endl;
		cerr << "obsolete and it will be removed in the near future." << endl;
		cerr << "The key point limit (--maxkey) is useful for large" << endl;
		cerr << "images with lots of features which could easily yield" << endl;
		cerr << "too many keypoints for a particular system. e.g. " << endl;
		cerr << "embedded systems. Please note that the feature extractor" << endl;
		cerr << "generates a larger number of key points than defined" << endl;
		cerr << "by this limit and only decides to retain that number" << endl;
		cerr << "in the end. Noisy images can be enhanced by denoising" << endl;
		cerr << "(--denoise)." << endl;
		cerr << visible;
		return 0;
	}

	showGui = vm.count("gui");
	autoAlign = vm.count("autoalign");
	srcScaling = vm.count("scaling");
	denoise = vm.count("denoise");
	radial = vm.count("radial");
	face = vm.count("face");
	distance = vm.count("distance");
#endif

#ifndef _WASM
	poppy::init(showGui, numberOfFrames, matchTolerance, contourSensitivity, maxKeypoints, autoAlign, radial, face, denoise, srcScaling, frameRate, pyramidLevels, fourcc);
	run(imageFiles, outputFile, phase, distance);
#else
	std::cerr << "Entering main loop..." << std::endl;
	std::cerr << "loaded" << std::endl;

	emscripten_set_main_loop(loop, 0, true);
	std::cerr << "Main loop canceled..." << std::endl;
#endif
}

extern "C" {

int load_images(char *file_path1, char *file_path2, double tolerance, bool face, bool autoscale, double numberOfFrames, bool autoalign) {
	try {
		std::vector<string> imageFiles;
		bool showGui = poppy::Settings::instance().show_gui;
		double frameRate = poppy::Settings::instance().frame_rate;
		size_t pyramidLevels = poppy::Settings::instance().pyramid_levels;
		double matchTolerance = tolerance;
		double contourSensitivity = poppy::Settings::instance().contour_sensitivity;
		off_t maxKeypoints = poppy::Settings::instance().max_keypoints;
		bool autoAlign = autoalign;
		bool radial = poppy::Settings::instance().enable_radial_mask;
		bool srcScaling = autoscale;
		bool denoise = false;
		string outputFile = "output.mkv";

		imageFiles.push_back(string(file_path1));
		imageFiles.push_back(string(file_path2));
		poppy::init(showGui, numberOfFrames, matchTolerance, contourSensitivity, maxKeypoints, autoAlign, radial, face, denoise, srcScaling, frameRate, pyramidLevels, "");
		std::thread t([=](){
				try {
				running = true;
				run(imageFiles, outputFile, -1, false);
				running = false;
				} catch (std::exception& ex) {
					std::cerr << "thread caught: " << ex.what() << std::endl;
					throw ex;
				} catch (...) {
					std::cerr << "thread caught" << std::endl;
				}
		});
		t.detach();
	} catch(std::exception& ex) {
		std::cerr << "caught: " << ex.what() << std::endl;
	} catch(...) {
		std::cerr << "caught" << std::endl;
	}
	return 0;
}
}
