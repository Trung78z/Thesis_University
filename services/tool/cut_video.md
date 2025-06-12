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




person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic lightdone