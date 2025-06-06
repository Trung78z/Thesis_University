ffmpeg -ss 00:05:00 -i video.mp4 -t 00:06:00 -c copy output.mp4


ffmpeg -ss 00:10:00 -i video.mp4 -c:v libx264 -c:a aac -preset fast output.mp4
