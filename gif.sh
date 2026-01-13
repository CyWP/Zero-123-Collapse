#!/bin/bash

# Usage: ./gif.sh <input_folder> <output.gif> <framerate>
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_folder> <output.gif> <framerate>"
    exit 1
fi

input_folder="$1"
output_gif="$2"
framerate="$3"

ffmpeg -y -framerate "$framerate" -pattern_type glob -i "$input_folder/*.png" "$output_gif"