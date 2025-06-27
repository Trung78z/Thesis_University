ffmpeg -ss 00:15:00 -i trip23.mp4 -t 00:10:00 -c copy /home/trung/Desktop/thesis/tool/trip_cut_1.mp4

ffmpeg -i input1.mp4 -ss 00:10:30 -to 00:30:00 -c:v copy -c:a copy -movflags +faststart output_1080p.mp4

ffmpeg -ss 00:10:30 -to 00:30:00 -i input1.mp4 \
-vf "scale=1920:1080" -b:v 2000k -c:v libx264 -preset fast \
-c:a aac -b:a 128k -movflags +faststart output_1080p.mp4


## Resize 
🔧 If You Want to Just Resize (Stretch/Distort)
ffmpeg -i input.mp4 -vf scale=320:240 -c:a copy output.mp4
✅ If You Want to Preserve Aspect Ratio (with Padding)
ffmpeg -i input.mp4 -vf "scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2" -c:a copy output.mp4

# # Extract frames at 10 FPS for training data
# ffmpeg -i input.mp4 -vf "fps=10" frame_%04d.png
# input1.mp4 = [ Driving Japan ] Tokyo City Highway. Relax and sleep.　首都高速

ffmpeg -ss 00:07:30 -i input1.mp4 -t 00:22:30 -an -vf "scale=1920:1080" -b:v 2000k -c:v libx264 -preset fast output.mp4



ffmpeg -ss 00:10:30 -to 00:30:00 -i test_video.mp4 -qscale:v 2 -vf "fps=0.5" images/car_%04d.jpg


ffmpeg -i test_video.mp4 -qscale:v 2 -vf "fps=2,scale=1280:720" images/thesis_%04d.jpg

ffmpeg -i -F-hrZKXM-k.mp4 -ss 00:10:30 -to 00:30:00 -c:v libx264 -c:a aac -movflags +faststart test_video.mp4

ffmpeg -hwaccel cuda -i -F-hrZKXM-k.mp4 -ss 00:10:30 -to 00:30:00 -c:v h264_nvenc -c:a aac -movflags +faststart test_video.mp4

# Cut image
ffmpeg -hwaccel cuda  -i "video/-F-hrZKXM-k.mp4" -q:v 1 -vf "fps=1/10" images_origin/frame_%04d.jpg

# Cut image with skip frame
ffmpeg -hwaccel cuda -i /home/trung/Desktop/Thesis_University/services/tool/video/test_video.mp4 \
-vf "select=gte(n\,4)" -vsync vfr -q:v 1 images/frame_%06d.jpg



✅ Một số kích thước nhỏ, tối ưu cho Jetson:
| Width x Height | Aspect Ratio | Ghi chú                                            |
| -------------- | ------------ | -------------------------------------------------- |
| **256x256**    | 1:1          | Nhỏ gọn, thường dùng trong training nhanh          |
| **320x240**    | 4:3          | Nhẹ, phù hợp với nhiều camera USB                  |
| **320x320**    | 1:1          | Rất phổ biến với YOLOv4-tiny, YOLOv5               |
| **416x416**    | 1:1          | Kích thước chuẩn của YOLOv3                        |
| **480x360**    | 4:3          | Cân bằng giữa chất lượng và tốc độ                 |
| **512x512**    | 1:1          | Chuẩn của nhiều model SSD                          |
| **640x480**    | 4:3          | Cũ nhưng phổ biến trong nhiều dataset (PASCAL VOC) |
| **640x640**    | 1:1          | Default của YOLOv5, YOLOv8                         |
| **672x384**    | 16:9         | Gần với camera ô tô, dashcam                       |
| **768x432**    | 16:9         | Cityscapes, dashcam, tầm trung                     |
| **800x600**    | 4:3          | Cũ, nhưng một số camera công nghiệp dùng           |
| **960x544**    | 16:9         | Được dùng trong một số hệ thống surveillance nhẹ   |



| Nhu cầu                              | Gợi ý kích thước          |
| ------------------------------------ | ------------------------- |
| **Tốc độ cực nhanh, Jetson Nano**    | 256x256, 320x240, 320x320 |
| **Cân bằng tốc độ + độ chính xác**   | 416x416, 512x512, 480x360 |
| **Chất lượng cao hơn (Xavier/Orin)** | 640x640, 768x432, 960x544 |
