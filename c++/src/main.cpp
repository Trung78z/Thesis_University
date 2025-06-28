#include <BYTETracker.h>
#include <lanevision/LaneDetector.h>

#include <cxxopts.hpp>
#include <header.hpp>
#include <iostream>
#include <string>
#include <utils.hpp>
#include <vector>

#include "Detect.h"
const int WIDTH = 1280;
const int HEIGHT = 720;
const int FPS = 30;

std::vector<int> trackClasses{0, 1, 2, 3, 5, 7};  // person, bicycle, car, motorcycle, bus, truck

bool isTrackingClass(int class_id) {
  for (auto &c : trackClasses) {
    if (class_id == c) return true;
  }
  return false;
}

/**
 * @brief Setting up Tensorrt logger
 */
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    // Only output logs with severity greater than warning
    if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
  }
} logger;

int main(int argc, char **argv) {
  try {
    cxxopts::Options options("test", "Run inference on a video or images (choose only one)");

    options.add_options()("v,video", "Video path", cxxopts::value<std::string>())(
        "i,images", "Images path", cxxopts::value<std::string>())(
        "m,engine", "Engine path", cxxopts::value<std::string>())("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    // Print usage instructions if --help is provided
    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      return 0;
    }

    std::string videoPath, imagePath, enginePath;

    // Check required parameters
    if (result.count("engine")) {
      enginePath = result["engine"].as<std::string>();
    } else {
      std::cerr << "❌ Error: --engine is required.\n";
      return 1;
    }

    if (result.count("video")) videoPath = result["video"].as<std::string>();
    if (result.count("images")) imagePath = result["images"].as<std::string>();

    // Ensure only one of --video or --images is provided
    if (!videoPath.empty() && !imagePath.empty()) {
      std::cerr << "❌ Error: Please provide either --video or --images, not both.\n";
      return 1;
    }

    if (videoPath.empty() && imagePath.empty()) {
      std::cerr << "❌ Error: You must provide either --video or --images.\n";
      return 1;
    }

    // Do not run if the file is .onnx
    if (enginePath.find(".onnx") != std::string::npos) {
      std::cout << "ℹ️ ONNX model detected, skipping inference.\n";
      return 0;
    }

    // Load model
    std::cout << "🔧 Loading engine from: " << enginePath << std::endl;
    Detect model(enginePath, logger);

    // Run inference based on input type
    if (!videoPath.empty()) {
      std::cout << "🎞️ Running video inference on: " << videoPath << std::endl;
      if (checkVideo(videoPath)) {
        return runVideo(videoPath, model);
      } else {
        std::cerr << "❌ Invalid video path.\n";
        return 1;
      }
    }

    if (!imagePath.empty()) {
      std::vector<std::string> imageList;
      std::cout << "🖼️ Running image inference in folder: " << imagePath << std::endl;
      if (checkImages(imagePath, imageList)) {
        return runImages(imageList, model);
      } else {
        std::cerr << "❌ No valid images found in: " << imagePath << std::endl;
        return 1;
      }
    }

  } catch (const std::exception &e) {
    std::cerr << "❌ Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

int runImages(vector<string> imagePathList, Detect model) {
  // Path to folder containing images
  for (const auto &imagePath : imagePathList) {
    // Open image
    Mat image = imread(imagePath);
    if (image.empty()) {
      cerr << "Error reading image: " << imagePath << endl;
      continue;
    }

    vector<Detection> objects;
    model.preprocess(image);

    auto start = std::chrono::system_clock::now();
    model.infer();
    auto end = std::chrono::system_clock::now();

    model.postprocess(objects);
    model.draw(image, objects);
    auto tc =
        (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    printf("cost %2.4lf ms\n", tc);

    model.draw(image, objects);

    imshow("Result", image);

    waitKey(0);
  }
  return 0;
}

int runVideo(const string path, Detect model) {
  cout << "Opening video: " << path << endl;
  // Example GStreamer pipeline for Jetson (commented out)
  // cv::VideoCapture('nvarguscamerasrc !
  // video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv !
  // video/x-raw,format=BGRx ! videoconvert ! appsink', cv::CAP_GSTREAMER);
  // std::string capture_pipeline =
  //     "nvarguscamerasrc ! "
  //     "video/x-raw(memory:NVMM), width=" + std::to_string(WIDTH) +
  //     ", height=" + std::to_string(HEIGHT) +
  //     ", format=NV12, framerate=" + std::to_string(FPS) + "/1 ! "
  //     "nvvidconv flip-method=0 ! "
  //     "video/x-raw, format=BGRx ! "
  //     "videoconvert ! "
  //     "video/x-raw, format=BGR ! "
  //     "appsink drop=true";
  // cv::VideoCapture cap(capture_pipeline, cv::CAP_GSTREAMER);
  cv::VideoCapture cap(path);

  if (!cap.isOpened()) {
    cerr << "Error: Cannot open video file!" << endl;
    return 0;
  }
  // Get frame width, height, and FPS
  double fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

  BYTETracker tracker(fps, 30);
  int frameCount = 0;
  auto fpsStartTime = std::chrono::steady_clock::now();
  int maxSpeed = -1;     // km/h
  int accMaxSpeed = 90;  // km/h
  int accSpeed = 60;     // km/h
  int total_ms = 0;
  LaneDetector laneDetector;
  while (true) {
    auto now = std::chrono::steady_clock::now();
    auto start = std::chrono::system_clock::now();
    cv::Mat image;
    cap >> image;

    if (image.empty()) {
      break;
    }
    // Resize the image to fit the window
    // cv::resize(image, image, cv::Size(WIDTH, HEIGHT));
    vector<Detection> res;
    const float ratio_h = model.getInputH() / (float)image.rows;
    const float ratio_w = model.getInputW() / (float)image.cols;
    model.preprocess(image);

    model.infer();

    model.postprocess(res);

    std::vector<Object> objects;
    for (const auto &obj : res) {
      auto box = obj.bbox;
      auto class_id = obj.class_id;
      auto conf = obj.conf;

      // Adjust box back to original image size
      if (ratio_h > ratio_w) {
        box.x = box.x / ratio_w;
        box.y = (box.y - (model.getInputH() - ratio_w * image.rows) / 2) / ratio_w;
        box.width = static_cast<int>(box.width / ratio_w);
        box.height = static_cast<int>(box.height / ratio_w);
      } else {
        box.x = (box.x - (model.getInputW() - ratio_h * image.cols) / 2) / ratio_h;
        box.y = box.y / ratio_h;
        box.width = static_cast<int>(box.width / ratio_h);
        box.height = static_cast<int>(box.height / ratio_h);
      }

      // Clamp box coordinates to image size
      box.x = std::max(0.0f, static_cast<float>(box.x));
      box.y = std::max(0.0f, static_cast<float>(box.y));
      box.width = std::min(static_cast<float>(box.width), static_cast<float>(image.cols - box.x));
      box.height = std::min(static_cast<float>(box.height), static_cast<float>(image.rows - box.y));

      if (isTrackingClass(class_id)) {
        Object obj{box, class_id, conf};
        objects.push_back(obj);
      }
    }

    // Tracking
    std::vector<STrack> output_stracks = tracker.update(objects);

    auto end = std::chrono::system_clock::now();
    total_ms =
        total_ms + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    for (int i = 0; i < output_stracks.size(); i++) {
      std::vector<float> tlwh = output_stracks[i].tlwh;
      // bool vertical = tlwh[2] / tlwh[3] > 1.6;
      // if (tlwh[2] * tlwh[3] > 20 && !vertical)
      if (tlwh[2] * tlwh[3] > 20) {
        cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
        cv::putText(image, cv::format("%d", output_stracks[i].track_id),
                    cv::Point(tlwh[0], tlwh[1] - 5), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::rectangle(image, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
      }
    }

    std::vector<cv::Vec4i> lanes = laneDetector.detectLanes(image);

    // Log detected objects
    for (const auto &obj : res) {
      auto box = obj.bbox;
      auto class_id = obj.class_id;
      auto conf = obj.conf;

      // Adjust box back to original image size
      if (ratio_h > ratio_w) {
        box.x = box.x / ratio_w;
        box.y = (box.y - (model.getInputH() - ratio_w * image.rows) / 2) / ratio_w;
        box.width = static_cast<int>(box.width / ratio_w);
        box.height = static_cast<int>(box.height / ratio_w);
      } else {
        box.x = (box.x - (model.getInputW() - ratio_h * image.cols) / 2) / ratio_h;
        box.y = box.y / ratio_h;
        box.width = static_cast<int>(box.width / ratio_h);
        box.height = static_cast<int>(box.height / ratio_h);
      }

      // Clamp box coordinates to image size
      box.x = std::max(0.0f, static_cast<float>(box.x));
      box.y = std::max(0.0f, static_cast<float>(box.y));
      box.width = std::min(static_cast<float>(box.width), static_cast<float>(image.cols - box.x));
      box.height = std::min(static_cast<float>(box.height), static_cast<float>(image.rows - box.y));
      cv::Point bottom_center(box.x + box.width / 2, box.y + box.height);
      if (lanes.size() >= 2) {
        // Get 2 lanes (left - right)
        cv::Vec4i l0 = lanes[0];
        cv::Vec4i l1 = lanes[1];

        // Calculate the middle point of the lane to draw a line
        cv::Point laneTop((l0[0] + l1[0]) / 2, (l0[1] + l1[1]) / 2);
        cv::Point laneBottom((l0[2] + l1[2]) / 2, (l0[3] + l1[3]) / 2);

        // Create a polygon for the lane area in order: top-left → top-right → bottom-right → bottom-left
        std::vector<cv::Point> lane_area = {
            cv::Point(l0[0], l0[1]),  // top-left
            cv::Point(l1[0], l1[1]),  // top-right
            cv::Point(l1[2], l1[3]),  // bottom-right
            cv::Point(l0[2], l0[3])   // bottom-left
        };

        if (obj.class_id == 2 || obj.class_id == 4 ||
            obj.class_id == 5)  // only consider car/bus/truck
        {
          if (cv::pointPolygonTest(lane_area, bottom_center, false) >= 0)  // vehicle is in the lane
          {
            // Draw a dot at the center of the bbox
            cv::Point mid(box.x + box.width / 2,
                          box.y + box.height / 2);                 // bbox center
            cv::circle(image, mid, 5, cv::Scalar(0, 255, 0), -1);  // green dot
          }
        }
      }

      if (class_id >= 12 && class_id <= 17 && conf > 0.9) {
        // Save frame with speed limit sign

        int newSpeed = (class_id - 9) * 10;  // class_id 12 -> 30km/h, 13 -> 40, etc.
        maxSpeed = newSpeed;
      }
    }

    model.draw(image, res);

    laneDetector.drawLanes(image, lanes);
    // FPS calculation
    frameCount++;

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - fpsStartTime).count();
    if (elapsed >= 1) {
      fps = frameCount / static_cast<double>(elapsed);
      frameCount = 0;
      fpsStartTime = now;
    }
    // Draw max speed
    if (maxSpeed != -1) {
      cv::putText(image, cv::format("Max Speed: %dKm/h", maxSpeed), cv::Point(10, 60),
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
    } else {
      cv::putText(image, "No Speed Limit Detected", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
                  0.7, cv::Scalar(255, 0, 0), 2);
    }

    // Draw Cruise Control
    cv::putText(image, cv::format("Max Cruise Control: %dKm/h", accMaxSpeed), cv::Point(10, 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    // Draw current speed
    cv::putText(image, cv::format("Control speed: %dKm/h", accSpeed), cv::Point(10, 120),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    if (maxSpeed != -1) {
      if (accSpeed < maxSpeed && accSpeed < accMaxSpeed) {
        accSpeed += 1;  // Increase speed by 1 km/h
      } else if (accSpeed > maxSpeed && accSpeed > 0) {
        accSpeed -= 1;  // Decrease speed by 1 km/h
      }
    } else {
      accSpeed = accMaxSpeed;  // Reset to max speed if no speed limit detected
    }
    // Draw FPS
    cv::putText(image, cv::format("FPS: %.2f", fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(0, 255, 0), 2);

    cv::imshow("Result", image);
    if (cv::waitKey(1) == 'q') {  // Press 'q' to exit
      break;
    }
  }

  // Release resources
  cap.release();
  cv::destroyAllWindows();
  return 0;
}