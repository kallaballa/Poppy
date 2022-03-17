#!/bin/bash

set -e
cd images
LOW=`mktemp --suffix=.mp4`
mkdir -p ../videos/
rm -f ../videos/*mkv

../src/poppy -g -f240 -o ../videos/faces.mkv some1.png some2.png some3.png some1.png
mencoder -vf scale=250:320 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 15 -ofps 6 ../videos/faces.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/faces.gif

../src/poppy -t0.6 -g -f240 -o ../videos/cars.mkv kindpng1s.png kindpng2s.png kindpng1s.png
mencoder -vf scale=250:160 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 15 -ofps 6 ../videos/cars.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/cars.gif

../src/poppy -t0.2 -d -g -f240 -o ../videos/browns.mkv amir1.jpg amir2.jpg amir1.jpg
mencoder -vf scale=360:202 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 15 -ofps 6 ../videos/browns.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/browns.gif

../src/poppy -t5 -g -f240 -o ../videos/catdog.mkv cat.png dog.png cat.png
mencoder -vf scale=320:320 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 15 -ofps 6 ../videos/catdog.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/catdog.gif

../src/poppy -t10 -g -f240 -o ../videos/squarecircle.mkv circle.png square.png circle.png
mencoder -vf scale=320:320 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 15 -ofps 6 ../videos/squarecircle.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/squarecircle.gif

../src/poppy -t10 -g -f240 -o ../videos/numbers.mkv 1st.png 2nd.png 3rd.png 2nd.png 1st.png
mencoder -vf scale=320:320 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 15 -ofps 6 ../videos/numbers.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/numbers.gif

../src/poppy -t5 -g -f240 -o ../videos/flowers.mkv flower1.png flower2.png flower1.png
mencoder -vf scale=320:320 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 15 -ofps 6 ../videos/flowers.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/flowers.gif

../src/poppy -g -f240 -o ../videos/yalefaces.mkv subject01*.png subject01.glasses.png
mencoder -vf scale=320:320 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 15 -ofps 6 ../videos/yalefaces.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/yalefaces.gif

rm "$LOW"
