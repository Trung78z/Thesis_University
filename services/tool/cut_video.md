ffmpeg -ss 00:15:00 -i trip23.mp4 -t 00:10:00 -c copy /home/trung/Desktop/thesis/tool/trip_cut_1.mp4

ffmpeg -i input1.mp4 -ss 00:10:30 -to 00:30:00 -c:v copy -c:a copy -movflags +faststart output_1080p.mp4

ffmpeg -ss 00:10:30 -to 00:30:00 -i input1.mp4 \
-vf "scale=1920:1080" -b:v 2000k -c:v libx264 -preset fast \
-c:a aac -b:a 128k -movflags +faststart output_1080p.mp4


# # Extract frames at 10 FPS for training data
# ffmpeg -i input.mp4 -vf "fps=10" frame_%04d.png
# input1.mp4 = [ Driving Japan ] Tokyo City Highway. Relax and sleep.　首都高速

ffmpeg -ss 00:07:30 -i input1.mp4 -t 00:22:30 -an -vf "scale=1920:1080" -b:v 2000k -c:v libx264 -preset fast output.mp4



ffmpeg -ss 00:10:30 -to 00:30:00 -i test_video.mp4 -qscale:v 2 -vf "fps=0.5" images/car_%04d.jpg


ffmpeg -i test_video.mp4 -qscale:v 2 -vf "fps=2,scale=1280:720" images/thesis_%04d.jpg

ffmpeg -i -F-hrZKXM-k.mp4 -ss 00:10:30 -to 00:30:00 -c:v libx264 -c:a aac -movflags +faststart test_video.mp4

ffmpeg -hwaccel cuda -i -F-hrZKXM-k.mp4 -ss 00:10:30 -to 00:30:00 -c:v h264_nvenc -c:a aac -movflags +faststart test_video.mp4

# Cut image
ffmpeg -hwaccel cuda  -i /home/trung/Desktop/Thesis_University/services/tool/video/test_video.mp4 -q:v 1 images/frame_%06d.jpg

# Cut image with skip frame
ffmpeg -hwaccel cuda -i /home/trung/Desktop/Thesis_University/services/tool/video/test_video.mp4 \
-vf "select=gte(n\,4)" -vsync vfr -q:v 1 images/frame_%06d.jpg
