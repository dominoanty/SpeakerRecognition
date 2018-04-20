#!/bin/bash

for i in *.mpeg;
do
	ffmpeg -i "$i" -acodec pcm_s16le -ac 1 -ar 16000 `basename "$i" .mpeg`.wav
done
for i in *.ogg;
do
	ffmpeg -i "$i" -acodec pcm_s16le -ac 1 -ar 16000 `basename "$i" .ogg`.wav
done
