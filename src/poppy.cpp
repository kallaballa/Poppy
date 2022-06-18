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
		if(gif_encoder != nullptr)
			GifWriteFrame(gif_encoder, bgra.data, bgra.size().width, bgra.size().height, 100.0/poppy::Settings::instance().frame_rate);
		Mat scaled;
		auto size = canvas->getSize();
		double w = size.first;
		double h = size.second;
		double sx = w / bgra.size().width;
		double sy = h / bgra.size().height;
		double factor = std::min(sx, sy);
		resize(bgra, scaled, Size(0, 0), factor, factor, INTER_CUBIC);
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

		if(!running) {
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
		}
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
			return Mat();
		}
		if(loadedSurface->format->BytesPerPixel == 1) {
			result = Mat(loadedSurface->h, loadedSurface->w, CV_8UC1, (unsigned char*) loadedSurface->pixels, loadedSurface->pitch);
			cvtColor(result,result, COLOR_GRAY2BGR);
		} else if(loadedSurface->format->BytesPerPixel == 3) {
			result = Mat(loadedSurface->h, loadedSurface->w, CV_8UC3, (unsigned char*) loadedSurface->pixels, loadedSurface->pitch);
			if(loadedSurface->format->Rmask == 0x0000ff)
				cvtColor(result,result, COLOR_RGB2BGR);
		} else if(loadedSurface->format->BytesPerPixel == 4) {
			result = Mat(loadedSurface->h, loadedSurface->w, CV_8UC4, (unsigned char*) loadedSurface->pixels, loadedSurface->pitch);
			if(loadedSurface->format->Rmask == 0x000000ff)
				cvtColor(result,result, COLOR_RGBA2BGR);
			else
				cvtColor(result,result, COLOR_RGBA2RGB);
		} else {
			std::cerr << "Unsupported image depth" << std::endl;
			return Mat();
		}
	}
	return result;
}

void run(const std::vector<string> &imageFiles, const string &outputFile, double phase, bool distance) {
	std::cerr << "usage0: " << (uintptr_t) sbrk(0) << std::endl;

	for (auto p : imageFiles) {
		if (!std::filesystem::exists(p))
			throw std::runtime_error("File doesn't exist: " + p);
	}

	Mat image1, denoise1;
	try {
		image1 = read_image(imageFiles[0]);
		if (poppy::Settings::instance().enable_denoise) {
			fastNlMeansDenoising(image1, denoise1, 10, 7, 21);
			denoise1.copyTo(image1);
		}
		if (image1.empty()) {
			std::cerr << "Can't read (invalid?) image file: " + imageFiles[0] << std::endl;
			exit(2);
		}
	} catch (std::exception& ex) {
		std::cerr << "Can't read (invalid?) image file: " + imageFiles[0] << " Exception: " << ex.what() << std::endl;
		exit(2);
	}

	Mat image2;

	Size szUnion(0, 0);
	for (size_t i = 0; i < imageFiles.size(); ++i) {
		Mat img = read_image(imageFiles[i]);
		if (szUnion.width < img.cols) {
			szUnion.width = img.cols;
		}

		if (szUnion.height < img.rows) {
			szUnion.height = img.rows;
		}
	}
	cerr << "union: " << szUnion << endl;
	Mat mUnion(szUnion.height, szUnion.width, image1.type(), { 0, 0, 0 });
	if (poppy::Settings::instance().enable_src_scaling) {
		Mat clone = image1.clone();
		resize(clone, image1, szUnion, INTER_CUBIC);
	} else {
		Rect centerRect((szUnion.width - image1.cols) / 2.0, (szUnion.height - image1.rows) / 2.0, image1.cols, image1.rows);
		image1.copyTo(mUnion(centerRect));
		image1 = mUnion.clone();
	}

#ifndef _WASM
	VideoWriter output(outputFile, VideoWriter::fourcc('F', 'F', 'V', '1'), poppy::Settings::instance().frame_rate, Size(szUnion.width, szUnion.height));

#else

	{
	std::unique_lock lock(frameBufferMtx);
		if(gif_encoder != nullptr)
			delete gif_encoder;

		if(canvas != nullptr)
			delete canvas;

		canvas = new poppy::Canvas(160, 160, false);
		gif_encoder = new GifWriter();
		GifBegin(gif_encoder, "current.gif", szUnion.width, szUnion.height, 100.0/poppy::Settings::instance().frame_rate);
	}
	ChannelWriter output;
#endif

	for (size_t i = 1; i < imageFiles.size(); ++i) {
		Mat image2, denoise2;
		try {
			image2 = read_image(imageFiles[i]);
			if (poppy::Settings::instance().enable_denoise) {
				std::cerr << "Denoising -> " << endl;
				fastNlMeansDenoising(image2, denoise2, 10, 7, 21);
				denoise2.copyTo(image2);
			}

			if (image2.empty()) {
				std::cerr << "Can't read (invalid?) image file: " + imageFiles[i] << std::endl;
				exit(2);
			}

			if (poppy::Settings::instance().enable_src_scaling) {
				Mat clone = image2.clone();
				resize(clone, image2, szUnion, INTER_CUBIC);
			} else {
				mUnion = Scalar::all(255);
				Rect cr((szUnion.width - image2.cols) / 2.0, (szUnion.height - image2.rows) / 2.0, image2.cols, image2.rows);
				image2.copyTo(mUnion(cr));
				image2 = mUnion.clone();
			}

			if (image1.cols != image2.cols || image1.rows != image2.rows) {
				std::cerr << "Image file sizes don't match: " << image1.size() << "/" << image2.size() << "/" << szUnion << std::endl;
				exit(3);
			}
		} catch (std::exception& ex) {
			std::cerr << "Can't read (invalid?) image file: " + imageFiles[0] << " Exception: " << ex.what() << std::endl;
			exit(2);
		}
		std::cerr << "matching: " << imageFiles[i - 1] << " -> " << imageFiles[i] << " ..." << std::endl;
		std::cerr << "usage1: " << (uintptr_t) sbrk(0) << std::endl;

		poppy::morph(image1, image2, phase, distance, output);
		image1 = image2.clone();
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
		cerr << "too many keypoints for a particular machine. e.g. " << endl;
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

	showGui = vm.count("gui");
	autoAlign = vm.count("autoalign");
	srcScaling = vm.count("scaling");
	denoise = vm.count("denoise");
	radial = vm.count("radial");
	face = vm.count("face");
	distance = vm.count("distance");
#endif

#ifndef _WASM
	poppy::init(showGui, numberOfFrames, matchTolerance, contourSensitivity, maxKeypoints, autoAlign, radial, face, denoise, srcScaling, frameRate);
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
		double matchTolerance = tolerance;
		double contourSensitivity = poppy::Settings::instance().contour_sensitivity;
		off_t maxKeypoints = 200;
		bool autoAlign = autoalign;
		bool radial = poppy::Settings::instance().enable_radial_mask;
		bool srcScaling = autoscale;
		bool denoise = false;
		string outputFile = "output.mkv";

		imageFiles.push_back(string(file_path1));
		imageFiles.push_back(string(file_path2));
		poppy::init(showGui, numberOfFrames, matchTolerance, contourSensitivity, maxKeypoints, autoAlign, radial, face, denoise, srcScaling, frameRate);
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
	} catch(...) {
//		std::cerr << "caught: " << ex.what() << std::endl;
		std::cerr << "caught" << std::endl;
	}
	return 0;
}
}
