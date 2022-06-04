# Poppy
An automatic and non-domain-specific image morphing tool that doesn't require manual keypoints.

This is very much a work in progress. And works best on similar objects from a similar perspective.
Please note that though the algorithm is capable of rotating and translating images to match (if they are similar enough), **sometimes you get the best result when you scale/rotate/translate the source images to match each other by hand**.

## Demos
The following demos are very low quality renderings of what Poppy can do. If you are truly curious, you should really try it yourself! It is quick and easy to do. Check out [this script](https://github.com/kallaballa/Poppy/blob/main/make_demos.sh) to see how the demos are generated. If you are too lazy for that there is also a [video](https://vimeo.com/679551761) containing all demos.

![Car morphing](https://github.com/kallaballa/Poppy/blob/main/demo/cars.gif?raw=true)
![Morphing face features](https://github.com/kallaballa/Poppy/blob/main/demo/browns.gif?raw=true)
![Morphing faces](https://github.com/kallaballa/Poppy/blob/main/demo/faces.gif?raw=true)
![Yale faces morphing](https://github.com/kallaballa/Poppy/blob/main/demo/yalefaces.gif?raw=true)
![Flower morphing to another flower](https://github.com/kallaballa/Poppy/blob/main/demo/flowers.gif?raw=true)
![Square morphing to a circle](https://github.com/kallaballa/Poppy/blob/main/demo/squarecircle.gif?raw=true)
![Morphing from one number to the next](https://github.com/kallaballa/Poppy/blob/main/demo/numbers.gif?raw=true)
![Morphing from a cat to a dog](https://github.com/kallaballa/Poppy/blob/main/demo/catdog.gif?raw=true)

(Yale face images taken from http://vision.ucsd.edu/content/yale-face-database)

## Dependencies
* libboost-program-options
* opencv4
* SDL
* SDL_Image

## Build instructions for Ubuntu 22.04

```bash
    apt install git-core build-essential libsdl-image1.2-dev libsdl1.2-dev libopencv-dev
    git clone https://github.com/kallaballa/Poppy.git
    cd Poppy
    make -j
```

## Usage

```
Usage: poppy [OPTIONS]... [IMAGEFILES]...
Poppy automatically creates smooth transitions of shape
and color between two images. That process is called 
image morphing and can be seen as a form of tweening or
interpolation.

Default options will work fine on good source material.
If you don't like the result you might try aligning the
source images by hand (instead of using --autoalign). 
Anyway, there are also a couple of options you can
specify. But usually you would only want to do this if
you either have bad source material, feel like
experimenting or are trying to do something funny.
The first thing to try is to adjust the match
tolerance (--tolerance). If you want to tinker more,
You could enable the gui (--gui) and play with the
tolerance and maybe a little with contour sensitivity
(--contour) and watch how it effects the algorithm.
You probably shouldn't waste much time on the contour
sensitivity parameter because it has little or even 
detrimental effect, which makes it virtually obsolete
and it will be removed in the near future.
The key point limit (--maxkey) is useful for large
images with lots of features which could easily yield
too many keypoints for a particular machine. e.g. 
embedded systems. Please note that the feature extractor
generates a larger number of key points than defined
by this limit and only decides to retain that number
in the end. The only means of reducing the number of
generated key points at the moment is to denoise
(--denoise) the source images. Obviously that is not
optimal because you have no control over which
features will be removed. Usually the parameter is used
to enhance noisy images.

Options:
  -g [ --gui ]                        Show analysis windows.
  -e [ --face ]                       Enable face detection mode. Use if your 
                                      source material consists of faces only.
  -r [ --radial ]                     Use a radial mask to emphasize features 
                                      in the center.
  -a [ --autoalign ]                  Try to automatically align (rotate and 
                                      translate) the source material to match.
  -d [ --denoise ]                    Denoise images before morphing.
  -n [ --distance ]                   Calculate the morph distance and return.
  -s [ --scaling ]                    Instead of extending the source images, 
                                      to match in size, use scaling.
  -m [ --maxkey ] arg (=-1)           Manual override for the number of 
                                      keypoints to retain during detection. The
                                      default is automatic determination of 
                                      that number.
  -b [ --rate ] arg (=30)             The frame rate of the output video.
  -f [ --frames ] arg (=60)           The number of frames to generate.
  -p [ --phase ] arg (=-1)            A value from 0 to 1 telling poppy how far
                                      into the morph to start from.
  -t [ --tolerance ] arg (=5)         How tolerant poppy is when matching 
                                      keypoints.
  -c [ --contour ] arg (=1)           How sensitive poppy is to contours.
  -o [ --outfile ] arg (=output.mkv)  The name of the video file to write to.
  -h [ --help ]                       Print the help message.
```

## Run

```bash
    cd Poppy
    src/poppy img1.png img2.png imgN.png
```

## Have fun!
