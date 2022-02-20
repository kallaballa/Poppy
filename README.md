# Poppy
An non-domain-specific morph algorithm that doesn't require manual keypoints

![Face morphing demo](https://github.com/kallaballa/Poppy/blob/main/demo/faces.gif?raw=true)
![Car morphing demo](https://github.com/kallaballa/Poppy/blob/main/demo/cars.gif?raw=true)
![Eye browns morphing demo](https://github.com/kallaballa/Poppy/blob/main/demo/browns.gif?raw=true)

This is very much a work in progress. And works best on similar objects from a similar perspective.
Please note that it isn't (yet) rotation or scale invariant, so you have to scale and rotate the source images to match each other.
## Dependencies
* boost
* libboost-program-options
* opencv4

## Build

```bash
    git clone https://github.com/kallaballa/Poppy.git
    cd Poppy
    make
```

## Usage
```
Usage: poppy [options] <imageFiles>...

Options:
  -f [ --frames ] arg (=60)             The number of frames to generate
  -l [ --lendev ] arg (=20)             The maximum length deviation in percent
                                        for the length test
  -a [ --angdev ] arg (=0.29999999999999999)
                                        The maximum angular deviation in 
                                        percent for the angle test
  -p [ --pairlen ] arg (=20)            The divider that controls the maximum 
                                        distance (diagonal/divider) for point 
                                        pairs
  -c [ --choplen ] arg (=2)             The interval in which traversal paths 
                                        (point pairs) are chopped
  -o [ --outfile ] arg (=output.mp4)    The name of the video file to write to
  -h [ --help ]                         Print help message

```

## Run

```bash
    cd Poppy
    src/poppy img1.png img2.png imgN.png
```

## Have fun!


