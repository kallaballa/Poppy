# Poppy
An automatic and non-domain-specific image morphing tool that doesn't require manual keypoints.

This is very much a work in progress. And works best on similar objects from a similar perspective.
Please note that though the algorithm is capable of rotating and translating images to match (if they are similar enough), **often you get the best result when you scale/rotate/translate the source images to match each other by hand**.

## Demos
The following demos are very low quality renderings of what Poppy can do. There are always two versions of a demo, one manually aligned and one auto-aligned. Auto-aligned in this case means that Poppy guesses the orientation and scale of the source images and automatically aligns them as best as possible. If you are truly curious, you should really try it yourself! It is quick and easy to do. Check out [this script](https://github.com/kallaballa/Poppy/blob/main/make_demos.sh) to see how the demos are generated. If you are too lazy for that there is also a [video](https://vimeo.com/679551761) containing all demos and of course the [web version](https://viel-zu.org/poppy/) (compiled with emscripten).

![Car morphing](https://github.com/kallaballa/Poppy/blob/main/demo/cars.gif?raw=true)
![Car morphing auto aligned](https://github.com/kallaballa/Poppy/blob/main/demo/cars-a.gif?raw=true)

![Morphing eye browns](https://github.com/kallaballa/Poppy/blob/main/demo/browns.gif?raw=true)
![Morphing eye browns auto aligned](https://github.com/kallaballa/Poppy/blob/main/demo/browns-a.gif?raw=true)

![Morphing faces](https://github.com/kallaballa/Poppy/blob/main/demo/faces.gif?raw=true)
![Morphing faces auto aligned](https://github.com/kallaballa/Poppy/blob/main/demo/faces-a.gif?raw=true)

![Yale faces morphing](https://github.com/kallaballa/Poppy/blob/main/demo/yalefaces.gif?raw=true)
![Yale faces morphing auto aligned](https://github.com/kallaballa/Poppy/blob/main/demo/yalefaces-a.gif?raw=true)

![Flower morphing to another flower](https://github.com/kallaballa/Poppy/blob/main/demo/flowers.gif?raw=true)
![Flower morphing to another flower auto aligne](https://github.com/kallaballa/Poppy/blob/main/demo/flowers-a.gif?raw=true)

![Square morphing to a circle](https://github.com/kallaballa/Poppy/blob/main/demo/squarecircle.gif?raw=true)
![Square morphing to a circle auto aligne](https://github.com/kallaballa/Poppy/blob/main/demo/squarecircle-a.gif?raw=true)

![Morphing from one number to the next](https://github.com/kallaballa/Poppy/blob/main/demo/numbers.gif?raw=true)
![Morphing from one number to the next auto aligne](https://github.com/kallaballa/Poppy/blob/main/demo/numbers-a.gif?raw=true)

![Morphing from a cat to a dog](https://github.com/kallaballa/Poppy/blob/main/demo/catdog.gif?raw=true)
![Morphing from a cat to a dog auto aligne](https://github.com/kallaballa/Poppy/blob/main/demo/catdog-a.gif?raw=true)

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
tolerance and watch how it effects the algorithm.
Noisy images can be enhanced by denoising (--denoise).
If you would like to tune how sensitive to faces poppy
is you should try the (--neighbors) parameter. 
Additionally you can influence quality of blending with
the --pyramid parameter. The deeper the pyramid the
better the quality of the blending (at the cost of 
performance).
The --fourcc parameter gives opportunity to select
which codec to use for the output file. --outfile
defines the path to the output file.

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
  -w [ --wait ]                       Wait at defined breakpoints for key 
                                      input. specifically the character q.
  -s [ --scaling ]                    Instead of extending the source images, 
                                      to match in size, use scaling.
  -b [ --rate ] arg (=30)             The frame rate of the output video.
  -i [ --neighbors ] arg (=6)         Face detection parameter, specifying how 
                                      many neighbors each candidate rectangle 
                                      should have to retain it.
  -f [ --frames ] arg (=60)           The number of frames to generate.
  -p [ --phase ] arg (=-1)            A value from 0 to 1 telling poppy how far
                                      into the morph to start from.
  -y [ --pyramid ] arg (=64)          How many levels to use for the laplacian 
                                      pyramid.
  -t [ --tolerance ] arg (=1)         How tolerant poppy is when matching 
                                      keypoints.
  -u [ --fourcc ] arg (=FFV1)         The four letter fourcc identifier 
                                      (https://en.wikipedia.org/wiki/FourCC) 
                                      which selects the video format. e.g: 
                                      "FFV1", "h264", "theo"
  -o [ --outfile ] arg (=output.mkv)  The name of the video file to write to.
  -h [ --help ]                       Print the help message.
```

## Run

```bash
    cd Poppy
    src/poppy img1.png img2.png imgN.png
```

## Have fun!
