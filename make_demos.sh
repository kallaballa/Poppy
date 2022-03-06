#!/bin/bash

set -e
cd images
LOW=`mktemp --suffix=.mp4`
mkdir -p ../videos/
rm ../videos/*mkv

../src/poppy -o ../videos/faces.mkv some1.png some2.png some3.png some1.png
mencoder -vf scale=250:320 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 30 -ofps 6 ../videos/faces.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/faces.gif

../src/poppy -o ../videos/cars.mkv kindpng1s.png kindpng2s.png kindpng1s.png
mencoder -vf scale=250:160 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 30 -ofps 6 ../videos/cars.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/cars.gif

../src/poppy -o ../videos/browns.mkv amir1.jpg amir2.jpg amir1.jpg
mencoder -vf scale=360:202 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 30 -ofps 6 ../videos/browns.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/browns.gif

../src/poppy -o ../videos/catdog.mkv -c1.5 -s2 cat.png dog.png cat.png
mencoder -vf scale=320:320 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 30 -ofps 6 ../videos/catdog.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/catdog.gif

../src/poppy -o ../videos/squarecircle.mkv -s8 circle.png square.png circle.png
mencoder -vf scale=320:320 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 30 -ofps 6 ../videos/squarecircle.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/squarecircle.gif

../src/poppy -o ../videos/numbers.mkv -s10 1st.png 2nd.png 3rd.png 2nd.png 1st.png
mencoder -vf scale=320:320 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 30 -ofps 6 ../videos/numbers.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/numbers.gif

../src/poppy -o ../videos/flowers.mkv -s1.5 flower1.jpg flower2.jpg flower1.jpg
mencoder -vf scale=320:320 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 30 -ofps 6 ../videos/flowers.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/flowers.gif

../src/poppy -o ../videos/yalefaces.mkv subject01*.png subject01.glasses.png
mencoder -vf scale=320:320 -ovc lavc -lavcopts vcodec=ffv1 -nosound -fps 30 -ofps 6 ../videos/yalefaces.mkv -o "$LOW"
ffmpeg -y -i "$LOW" ../demo/yalefaces.gif

rm "$LOW"
