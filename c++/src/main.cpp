#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "LaneDetector.hpp"
#include <chrono>
#include <iostream>

int main() {
    // Load the YOLO model
    cv::dnn::Net net = cv::dnn::readNet("runs/detect/train/weights/best.onnx");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // Open video file
    std::string video_path = "tool/output.mp4";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }

    // Lane detector configuration
    LaneDetector::Config video_config;
    video_config.cannyThresholds = cv::Vec2i(70, 170);
    video_config.houghThreshold = 50;
    video_config.minLineLength = 30;
    video_config.yBottomRatio = 0.95f;
    video_config.yTopRatio = 0.75f;
    video_config.slopeThreshold = 0.4f;
    video_config.x1 = 200;
    video_config.x2 = 1200;
    video_config.x3 = 550;
    video_config.x4 = 680;
    video_config.y = 400;

    LaneDetector detector(video_config);

    // FPS calculation variables
    float fps = 0;
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Resize frame
        cv::resize(frame, frame, cv::Size(1280, 720));

        // Calculate FPS
        frame_count++;
        if (frame_count % 10 == 0) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            fps = 10000.0f / duration.count();  // 10 frames / time in seconds
            start_time = end_time;
        }

        // Put FPS text on frame
        cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

        // YOLO detection (simplified - actual YOLOv8 inference would be more complex)
        // Note: This is a placeholder - you'll need to implement proper YOLOv8 inference
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(640, 640), cv::Scalar(0,0,0), true, false);
        net.setInput(blob);
        cv::Mat outputs = net.forward();

        // Process outputs (this part would need to be adapted to your specific YOLO model)
        // Here you would parse the outputs and draw bounding boxes
        // For now, we'll just use the original frame as annotated_frame
        cv::Mat annotated_frame = frame.clone();

        // Lane detection
        auto processed = detector.processImage(annotated_frame);
        annotated_frame = processed.first;

        // Show the frame
        cv::imshow("YOLOv8 Detection", annotated_frame);

        // Press 'q' to quit
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release resources
    cap.release();
    cv::destroyAllWindows();

    return 0;
}