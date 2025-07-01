#include <EgoVehicle.h>
#include <process.h>

#include <algorithm>
#include <chrono>
#include <deque>
#include <iomanip>
#include <map>

using namespace Config;

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
        // --- Ego Vehicle Control Variables ---
        float currentEgoSpeed = initialSpeedKph;
        double lastSpeedUpdateTime = 0;
        std::deque<float> speedChangeHistory;
        std::deque<float> distanceHistory;

        // --- Object tracking buffers ---
        std::map<int, std::deque<float>> objectBuffers;
        std::map<int, float> prevDistances;
        std::map<int, double> prevTimes;
        std::map<int, float> smoothedSpeeds;

        cxxopts::Options options("test", "Run inference on a video or images (choose only one)");

        options.add_options()("v,video", "Video path", cxxopts::value<std::string>())(
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

        std::cout << "Opening video: " << videoPath << std::endl;
        cv::VideoCapture cap(videoPath);

        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video file!" << std::endl;
            return 0;
        }
        // Get frame width, height, and fps
        double fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

        BYTETracker tracker(fps, 30);

        while (cap.isOpened()) {
            auto now = std::chrono::steady_clock::now();
            auto start = std::chrono::system_clock::now();
            cv::Mat image;
            cap >> image;

            if (image.empty()) {
                break;
            }

            double timeStart = getCurrentTimeInSeconds();

            // Resize the image to fit the window
            // cv::resize(image, image, cv::Size(width, height));
            std::vector<Detection> res;

            model.preprocess(image);
            model.infer();
            model.postprocess(image, res);

            std::vector<Object> objects = filterDetections(res);

            // Tracking
            std::vector<STrack> outputStracks = tracker.update(objects);

            // Draw center zone
            // cv::rectangle(image, cv::Point(xMin, 0), cv::Point(xMax, image.rows),
            // cv::Scalar(255, 255, 0), 2);

            // --- Ego Vehicle Speed Control Logic ---
            int targetId = -1;
            float maxHeight = 0;
            cv::Rect bestBox;

            // Process tracked detections
            selectTarget(outputStracks, xMin, xMax, targetId, bestBox, maxHeight);

            // Speed control logic
            std::string action = "FREE DRIVE";
            cv::Scalar actionColor = cv::Scalar(0, 255, 0);
            float avgDistance = 0.0f;
            float frontAbsoluteSpeed = 0.0f;
            updateSpeedControl(timeStart, targetId, bestBox, currentEgoSpeed, lastSpeedUpdateTime,
                               objectBuffers, prevDistances, prevTimes, smoothedSpeeds,
                               speedChangeHistory, avgDistance, frontAbsoluteSpeed, action,
                               actionColor);

            // Always display information on frame
            if (targetId != -1 && avgDistance > 0) {
                cv::putText(
                    image, "Distance: " + std::to_string(static_cast<int>(avgDistance)) + " m",
                    cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
                cv::putText(image,
                            "Front Speed: " + std::to_string(static_cast<int>(frontAbsoluteSpeed)) +
                                " km/h",
                            cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0),
                            2);

                // Draw distance zones indicator
                cv::Scalar zoneColor;
                if (avgDistance < criticalDistance) {
                    zoneColor = cv::Scalar(0, 0, 255);  // Red - emergency
                } else if (avgDistance < minFollowingDistance) {
                    zoneColor = cv::Scalar(0, 100, 255);  // Orange - danger
                } else if (avgDistance < targetFollowingDistance) {
                    zoneColor = cv::Scalar(0, 255, 255);  // Yellow - caution
                } else {
                    zoneColor = cv::Scalar(0, 255, 0);  // Green - safe
                }

                cv::circle(image, cv::Point(image.cols - 100, 100), 30, zoneColor, -1);
                cv::putText(image, std::to_string(static_cast<int>(avgDistance)) + "m",
                            cv::Point(image.cols - 120, 110), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 255, 255), 1);
            } else {
                cv::putText(image, "Distance: -- m", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX,
                            0.8, cv::Scalar(128, 128, 128), 2);
                cv::putText(image, "Front Speed: -- km/h", cv::Point(20, 80),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(128, 128, 128), 2);
            }

            // Always show ego speed and action
            cv::putText(
                image, "Ego Speed: " + std::to_string(static_cast<int>(currentEgoSpeed)) + " km/h",
                cv::Point(20, 120), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            cv::putText(image, "Action: " + action, cv::Point(20, 160), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, actionColor, 2);

            model.draw(image, outputStracks);

            cv::imshow("Improved Adaptive Speed Control", image);
            if (cv::waitKey(1) == 'q') {  // Press 'q' to exit
                break;
            }
        }

        // Release resources
        cap.release();
        cv::destroyAllWindows();
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}