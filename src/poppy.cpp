#include "poppy.hpp"
#include "canvas.hpp"
#include "util.hpp"


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

void run(const std::vector<string> &imageFiles, const string &outputFile, double phase, bool distance, bool buildUnion) {
	for (auto p : imageFiles) {
		if (!std::filesystem::exists(p))
			throw std::runtime_error("File doesn't exist: " + p);
	}

	Mat img1, denoise1;
	Mat corrected1, corrected2;
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

	Size szUnion(0, 0);

	if(buildUnion) {
		cerr << "building union" << endl;

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
	} else {
		szUnion.width = img1.cols;
		szUnion.height = img1.rows;
		size_t i = 1;
		for (; i < imageFiles.size(); ++i) {
			Mat img = read_image(imageFiles[i]);
			if(img.cols == img1.cols && img.rows == img1.rows)
				break;
			if (szUnion.width < img.cols) {
				szUnion.width = img.cols;
			}

			if (szUnion.height < img.rows) {
				szUnion.height = img.rows;
			}
			img.release();
		}
		cerr << "infering union from " + to_string(i) + " image(s)..." << endl;
	}


	if (poppy::Settings::instance().enable_src_scaling) {
		cerr << "scaling first image to: " << preserveAspect(img1.size(), szUnion) << "..." << endl;

		Mat clone = img1.clone();
		resize(clone, img1, preserveAspect(img1.size(), szUnion), INTER_LINEAR);
	}
	Mat mUnion(szUnion.height, szUnion.width, img1.type(), { 0, 0, 0 });
	mUnion = Scalar::all(0);

	if(phase == 0 || phase == 1) {
		double dx = fabs(img1.cols - szUnion.width) / 2.0;
		double dy = fabs(img1.rows - szUnion.height) / 2.0;
		Rect roi(dx, dy, img1.cols, img1.rows);
		img1.copyTo(mUnion(roi));
	} else {
		poppy::blur_margin(img1, szUnion, mUnion);
	}

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
	for (size_t i = 1; i < imageFiles.size(); ++i) {
		Mat img2, denoise2;
		try {
			img2 = read_image(imageFiles[i]);
			if (poppy::Settings::instance().enable_denoise) {
				std::cerr << "Denoising -> " << endl;
				fastNlMeansDenoising(img2, denoise2, 10, 7, 21);
				denoise2.copyTo(img2);
			}

			if (img2.empty()) {
				std::cerr << "Can't read (invalid?) image file: " + imageFiles[i] << std::endl;
				exit(2);
			}

			if (poppy::Settings::instance().enable_src_scaling) {
				Mat bg = Mat::zeros(szUnion, img2.type());
				Mat clone = img2.clone();
				cerr << "scaling image " << (i + 1) << " to: " << preserveAspect(img2.size(), szUnion) << "..." << endl;

				Size aspect = preserveAspect(img2.size(), szUnion);
				resize(clone, img2, aspect, INTER_LINEAR);


				if(phase == 0 || phase == 1) {
					double dx = fabs(img2.cols - szUnion.width) / 2.0;
					double dy = fabs(img2.rows - szUnion.height) / 2.0;
					Rect roi(dx, dy, img2.cols, img2.rows);
					img2.copyTo(bg(roi));
				} else {
					poppy::blur_margin(img2, szUnion, bg);
				}
				img2 = bg.clone();
//				poppy::show_image("aspect", img2);
			} else {
				if(phase == 0 || phase == 1) {
					double dx = fabs(img2.cols - szUnion.width) / 2.0;
					double dy = fabs(img2.rows - szUnion.height) / 2.0;
					Rect roi(dx, dy, img2.cols, img2.rows);
					img2.copyTo(mUnion(roi));
				} else {
					poppy::blur_margin(img2, szUnion, mUnion);
				}
				img2 = mUnion.clone();
//				poppy::show_image("mu2", img2);
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

		bool savedefd = poppy::Settings::instance().enable_face_detection;
		poppy::morph(img1, img2, corrected1, corrected2, phase, distance, output);
		poppy::Settings::instance().enable_face_detection = savedefd;
		img1 = corrected2.clone();
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
	bool enableWait = poppy::Settings::instance().enable_wait;
	size_t numberOfFrames = poppy::Settings::instance().number_of_frames;
	size_t pyramidLevels = poppy::Settings::instance().pyramid_levels;
	size_t maxKey = poppy::Settings::instance().max_keypoints;
	double frameRate = poppy::Settings::instance().frame_rate;
	double phase = -1;
	double matchTolerance = poppy::Settings::instance().match_tolerance;
	bool autoAlign = poppy::Settings::instance().enable_auto_align;
	bool radial = poppy::Settings::instance().enable_radial_mask;
	bool face = poppy::Settings::instance().enable_face_detection;
	bool srcScaling = true;
	bool denoise = poppy::Settings::instance().enable_denoise;
	size_t faceNeighbors = poppy::Settings::instance().face_neighbors;
	bool distance = false;
	bool noBuild = false;

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
	("wait,w", "Wait at defined breakpoints for key input. Specifically the character q.")
	("scaling,s", "Instead of extending the source images, to match in size, use scaling.")
	("no-build,l", "Don't read all images to build the dimension union. Instead use the dimensions of the first images until Poppy encounters the first dimensions again.")
	("rate,b", po::value<double>(&frameRate)->default_value(frameRate), "The frame rate of the output video.")
	("maxkey,m", po::value<size_t>(&maxKey)->default_value(maxKey), "The maximum number of keypoints to retain. Effects performance as well as quality. Often less is better and the default value probably is just fine.")
	("neighbors,i", po::value<size_t>(&faceNeighbors)->default_value(faceNeighbors), "Face detection parameter, specifying how many neighbors each candidate rectangle should have to retain it.")
	("frames,f", po::value<size_t>(&numberOfFrames)->default_value(numberOfFrames), "The number of frames to generate.")
	("phase,p", po::value<double>(&phase)->default_value(phase), "A value from 0 to 1 telling poppy how far into the morph to start from.")
	("pyramid,y", po::value<size_t>(&pyramidLevels)->default_value(pyramidLevels), "How many levels to use for the laplacian pyramid.")
	("tolerance,t", po::value<double>(&matchTolerance)->default_value(matchTolerance), "How tolerant poppy is when matching keypoints.")
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
		cerr << "tolerance and watch how it effects the algorithm." << endl;
		cerr << "Additionally you can try to play with the maximum" << endl;
		cerr << "number of retained key points but that will improve" << endl;
		cerr << "quality on very large images and in rare cases only." << endl;
		cerr << "Noisy images can be enhanced by denoising (--denoise)." << endl;
		cerr << "If you would like to tune how sensitive to faces poppy" << endl;
		cerr << "is you should try the (--neighbors) parameter. " << endl;
		cerr << "Additionally you can influence quality of blending with" << endl;
		cerr << "the --pyramid parameter. The deeper the pyramid the" << endl;
		cerr << "better the quality of the blending (at the cost of " << endl;
		cerr << "performance)." << endl;
		cerr << "The --fourcc parameter gives opportunity to select" << endl;
		cerr << "which codec to use for the output file. --outfile" << endl;
		cerr << "defines the path to the output file." << endl;
		cerr << visible;
		return 0;
	}

	showGui = vm.count("gui");
	enableWait = vm.count("wait");
	autoAlign = vm.count("autoalign");
	srcScaling = vm.count("scaling");
	denoise = vm.count("denoise");
	radial = vm.count("radial");
	face = vm.count("face");
	distance = vm.count("distance");
	noBuild = vm.count("no-build");
#endif

#ifndef _WASM
	poppy::init(showGui, numberOfFrames, matchTolerance, autoAlign, radial, face, denoise, srcScaling, frameRate, pyramidLevels, fourcc, enableWait, faceNeighbors);
	run(imageFiles, outputFile, phase, distance, !noBuild);
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
		bool autoAlign = autoalign;
		bool radial = poppy::Settings::instance().enable_radial_mask;
		bool srcScaling = autoscale;
		bool denoise = false;
		bool buildUnion = true;
		string outputFile = "output.mkv";

		imageFiles.push_back(string(file_path1));
		imageFiles.push_back(string(file_path2));
		poppy::init(showGui, numberOfFrames, matchTolerance, autoAlign, radial, face, denoise, srcScaling, frameRate, pyramidLevels, string(""), false, 6);
		std::thread t([=](){
				try {
				running = true;
				run(imageFiles, outputFile, -1, false, buildUnion);
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
