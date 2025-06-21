#!/bin/bash

INPUT=$1
OUT_DIR="images_origin"

mkdir -p $OUT_DIR

# Đếm số ảnh hiện có
n=$(ls $OUT_DIR | wc -l)
start_num=$((n+1))

# Chạy ffmpeg
ffmpeg -hwaccel cuda -i "$INPUT" -qscale:v 2 -vf "fps=1/3" -start_number $start_num $OUT_DIR/frame_%04d.jpg
