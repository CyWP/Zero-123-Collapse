#!/bin/bash

# Usage: ./video.sh <input_folder> <output.mp4> <framerate>
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_folder> <output.mp4> <framerate>"
    exit 1
fi

input_folder="$1"
output_video="$2"
framerate="$3"

ffmpeg -y -framerate "$framerate" -pattern_type glob -i "$input_folder/*.png" -c:v libx264 -pix_fmt yuv420p "$output_video"