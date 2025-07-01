#include <EgoVehicle.h>
#include <process.h>

#include <algorithm>
#include <chrono>
#include <deque>
#include <iomanip>
#include <map>
#include <numeric>

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

int main(int argc, char **argv) {
    try {
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
        // Get frame width, height, and FPS
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

            double t0 = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count() /
                        1000.0;

            // Resize the image to fit the window
            // cv::resize(image, image, cv::Size(width, height));
            std::vector<Detection> res;

            model.preprocess(image);
            model.infer();
            model.postprocess(image, res);

            std::vector<Object> objects;
            for (const auto &obj : res) {
                auto box = obj.bbox;
                auto classId = obj.classId;
                auto conf = obj.conf;

                if (isTrackingClass(classId)) {
                    Object obj{box, classId, conf};
                    objects.push_back(obj);
                }
            }

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
            for (const auto &track : outputStracks) {
                if (!track.is_activated) continue;

                // Get bounding box from tlbr (top-left, bottom-right format)
                const auto &tlbr = track.tlbr;
                float x1 = tlbr[0], y1 = tlbr[1], x2 = tlbr[2], y2 = tlbr[3];
                float xCenter = (x1 + x2) / 2.0f;

                if (xMin <= xCenter && xCenter <= xMax) {
                    float h = y2 - y1;  // height = bottom - top
                    if (h > maxHeight) {
                        maxHeight = h;
                        targetId = track.track_id;
                        bestBox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
                    }
                }
            }

            // Speed control logic
            std::string action = "FREE DRIVE";
            cv::Scalar actionColor = cv::Scalar(0, 255, 0);
            float avgDistance = 0.0f;
            float frontAbsoluteSpeed = 0.0f;

            if (targetId != -1 && maxHeight > 0) {
                float h = static_cast<float>(bestBox.height);
                float distance = (realObjectWidth * focalLength) / h;

                // Initialize buffers for new target
                if (objectBuffers.find(targetId) == objectBuffers.end()) {
                    objectBuffers[targetId] = std::deque<float>();
                    prevDistances[targetId] = distance;
                    prevTimes[targetId] = t0;
                    smoothedSpeeds[targetId] = 0.0f;
                }

                // Add distance to buffer
                objectBuffers[targetId].push_back(distance);
                if (objectBuffers[targetId].size() > 5) {
                    objectBuffers[targetId].pop_front();
                }

                // Calculate smoothed distance
                if (objectBuffers[targetId].size() >= 3) {
                    std::vector<float> sortedDistances(objectBuffers[targetId].begin(),
                                                       objectBuffers[targetId].end());
                    std::sort(sortedDistances.begin(), sortedDistances.end());
                    avgDistance = sortedDistances[sortedDistances.size() / 2];
                } else {
                    avgDistance = std::accumulate(objectBuffers[targetId].begin(),
                                                  objectBuffers[targetId].end(), 0.0f) /
                                  objectBuffers[targetId].size();
                }

                double dt = t0 - prevTimes[targetId];

                // Calculate relative speed
                if (dt >= minTimeDelta) {
                    float prevDistance = prevDistances[targetId];
                    float dDist = prevDistance - avgDistance;

                    if (std::abs(dDist) >= minDistDelta) {
                        float speedMps = dDist / dt;
                        float relativeSpeedKph = speedMps * 3.6f;

                        // Smooth the relative speed
                        smoothedSpeeds[targetId] =
                            smoothingFactor * relativeSpeedKph +
                            (1.0f - smoothingFactor) * smoothedSpeeds[targetId];

                        prevDistances[targetId] = avgDistance;
                        prevTimes[targetId] = t0;
                    }
                }

                // Calculate front vehicle absolute speed
                float relativeSpeed = smoothedSpeeds[targetId];
                frontAbsoluteSpeed = currentEgoSpeed - relativeSpeed;

                // Update ego speed if enough time has passed
                double speedUpdateDt = t0 - lastSpeedUpdateTime;
                if (speedUpdateDt >= speedUpdateInterval) {
                    // Determine driving state
                    auto [drivingState, urgencyLevel] =
                        getDrivingState(avgDistance, frontAbsoluteSpeed, currentEgoSpeed);

                    // Calculate target speed
                    float targetSpeed =
                        calculateTargetSpeed(avgDistance, frontAbsoluteSpeed, currentEgoSpeed,
                                             drivingState, urgencyLevel);

                    // Update ego speed smoothly
                    float oldEgoSpeed = currentEgoSpeed;
                    currentEgoSpeed = updateEgoSpeedSmooth(currentEgoSpeed, targetSpeed,
                                                           urgencyLevel, speedUpdateDt);

                    // Track speed change
                    float speedChange = currentEgoSpeed - oldEgoSpeed;
                    speedChangeHistory.push_back(speedChange);
                    if (speedChangeHistory.size() > 10) {
                        speedChangeHistory.pop_front();
                    }

                    lastSpeedUpdateTime = t0;

                    // Get action and color for display
                    getActionAndColor(drivingState, speedChange, action, actionColor);

                    std::cout << "[+] ID " << targetId << " | Dist: " << std::fixed
                              << std::setprecision(1) << avgDistance << "m | "
                              << "Front: " << frontAbsoluteSpeed << " km/h | "
                              << "Ego: " << currentEgoSpeed << " km/h | "
                              << "State: " << drivingState << " | Action: " << action << std::endl;
                }
            } else {
                // No target vehicle - free driving
                double speedUpdateDt = t0 - lastSpeedUpdateTime;
                if (speedUpdateDt >= speedUpdateInterval) {
                    // Gradually return to cruise speed
                    if (std::abs(currentEgoSpeed - cruiseSpeedKph) > 1) {
                        if (currentEgoSpeed < cruiseSpeedKph) {
                            currentEgoSpeed =
                                std::min(cruiseSpeedKph, currentEgoSpeed + gentleAdjustment);
                        } else {
                            currentEgoSpeed =
                                std::max(cruiseSpeedKph, currentEgoSpeed - gentleAdjustment);
                        }
                    }
                    lastSpeedUpdateTime = t0;
                }
            }

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