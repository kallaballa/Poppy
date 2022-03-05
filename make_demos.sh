#!/bin/bash

src/poppy -o yalefaces.mkv -f60 yalefacesout/subject01*.png yalefacesout/subject01.glasses.png
mencoder -vf scale=160:160 -ovc x264 -nosound -fps 30 -ofps 6 yalefaces.mkv -o low.mp4
ffmpeg -y -i low.mp4 demo/yalefaces.gif

src/poppy -o faces.mkv -f60 some1.png some2.png some3.png some1.png
mencoder -vf scale=250:320 -ovc x264 -nosound -fps 30 -ofps 6 faces.mkv -o low.mp4
ffmpeg -y -i low.mp4 demo/faces.gif

src/poppy -o cars.mkv -f60 kindpng1s.png kindpng2s.png kindpng1s.png
mencoder -vf scale=250:160 -ovc x264 -nosound -fps 30 -ofps 6 cars.mkv -o low.mp4
ffmpeg -y -i low.mp4 demo/cars.gif

src/poppy -o browns.mkv -f60 amir1.jpg amir2.jpg amir1.jpg
mencoder -vf scale=360:202 -ovc x264 -nosound -fps 30 -ofps 6 browns.mkv -o low.mp4
ffmpeg -y -i low.mp4 demo/browns.gif

src/poppy -s1.1 -g -o catdog.mkv -f60 cat.png dog.png cat.png
mencoder -vf scale=450:300 -ovc x264 -nosound -fps 30 -ofps 6 catdog.mkv -o low.mp4
ffmpeg -y -i low.mp4 demo/catdog.gif

