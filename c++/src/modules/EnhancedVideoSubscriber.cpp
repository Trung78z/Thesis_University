#include "modules/EnhancedVideoSubscriber.h"
#include <iomanip>

#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
using namespace std;
using namespace Config;
EnhancedVideoSubscriber::EnhancedVideoSubscriber(const string &model_path_)
    : model_(model_path_, Logger::getInstance()), laneDetector_(),
      tracker_(30.0, 30), frameCount_(0), fps_(30.0), maxSpeed_(-1),
      accSpeed_(config.speedControl.cruiseSpeedKph), noSpeedLimitFrames_(0),
      speedLimitStabilizer_(6),
      currentEgoSpeed_(config.speedControl.initialSpeedKph),
      enable_debug_output_(true), lastSpeedUpdateTime_(0), targetId_(-1),
      classId_(-1), lostTargetCount_(0), framesCurrentTargetOutsideLane_(0) {
    fpsStartTime_ = chrono::steady_clock::now();
    initializeEnhancedFeatures();
}

EnhancedVideoSubscriber::~EnhancedVideoSubscriber() { cv::destroyAllWindows(); }

void EnhancedVideoSubscriber::initializeEnhancedFeatures() {

    processing_times_.clear();
    total_detections_ = 0;
    emergency_stop_ = false;

    // Not using ros
    last_detection_time_ =
        std::chrono::steady_clock::now().time_since_epoch().count() / 1e9;
}

void EnhancedVideoSubscriber::checkSafetyConditions() {
    double current_time = getCurrentTimeInSeconds();

    if (current_time - last_detection_time_ > DETECTION_TIMEOUT) {
        if (!emergency_stop_) {
            emergency_stop_ = true;
            printf("Emergency stop activated: No detections for %.1f seconds\n",
                   current_time - last_detection_time_);
        }
    } else {
        emergency_stop_ = false;
    }

    if (currentEgoSpeed_ > maxSpeed_ * 1.2 && maxSpeed_ > 0) {
        // ROS_WARN("Speed limit exceeded: %.1f km/h (limit: %d km/h)",
        //          currentEgoSpeed_, maxSpeed_);
    }
}

void EnhancedVideoSubscriber::imageCallback(const cv::Mat &msg) {

    cv::Mat image = msg;

    if (image.empty()) {
        printf("Received empty image\n");
        return;
    }

    if (!videoRecorder_.isInitialized()) {
        videoRecorder_.init(image, "/home/trung/Desktop");
    }
    // ROS_INFO("Output file in %s", output_dir_.c_str());

    processEnhancedFrame(image);
    videoRecorder_.writeFrame(image);
}

void EnhancedVideoSubscriber::processEnhancedFrame(cv::Mat &image) {
    auto start = chrono::high_resolution_clock::now();
    double timeStart = getCurrentTimeInSeconds();

    checkSafetyConditions();

    std::vector<Detection> res;
    model_.preprocess(image);
    model_.infer();
    model_.postProcess(image, res);

    std::vector<Object> objects = filterDetections(res);
    std::vector<cv::Vec4i> lanes = laneDetector_.detectLanes(image);

    if (!objects.empty()) {
        last_detection_time_ = getCurrentTimeInSeconds();
    }

    std::vector<STrack> outputStracks = tracker_.update(objects);
    total_detections_ += outputStracks.size();

    performEnhancedTargetSelection(outputStracks, lanes, image);

    cv::Scalar actionColor = cv::Scalar(0, 255, 0);
    float avgDistance = 0.0f;
    float frontAbsoluteSpeed = 0.0f;
    bool accActive = classId_ == 2 ? true : false;
    egoVehicle_.updateSpeedControl(
        timeStart, targetId_, bestBox_, currentEgoSpeed_, lastSpeedUpdateTime_,
        objectBuffers_, prevDistances_, prevTimes_, smoothedSpeeds_,
        speedChangeHistory_, avgDistance, frontAbsoluteSpeed, actionColor);

    updateSpeedLimits(outputStracks);

    model_.draw(image, outputStracks);
    laneDetector_.drawLanes(image, lanes);

    updateFPS();

    hudRenderer_.setEmergencyStop(emergency_stop_);
    hudRenderer_.render(
        image, currentEgoSpeed_, accSpeed_, frontAbsoluteSpeed, avgDistance,
        accActive, egoVehicle_.getAction(), actionColor, fps_, targetId_,
        egoVehicle_.getEngineForce(), egoVehicle_.getThrottleForce(),
        egoVehicle_.getBrakeForce());

    if (enable_data_logging_) {
        auto end = chrono::high_resolution_clock::now();
        double processing_time = chrono::duration<double>(end - start).count();
    }

    if (enable_debug_output_) {
        cv::imshow("Enhanced Driving Assistant", image);
        cv::waitKey(1);
    }
}

void EnhancedVideoSubscriber::performEnhancedTargetSelection(
    const std::vector<STrack> &outputStracks,
    const std::vector<cv::Vec4i> &lanes, cv::Mat &image) {
    int detectedTargetId = -1;
    int detectedClassId = -1;
    cv::Rect bestBoxTmp;
    float maxBottomY = -1;
    float maxConfidence = 0;
    bool currentTargetStillExists = false;
    bool currentTargetInLane = false;
    float currentTargetBottomY = -1;

    if (targetId_ != -1) {
        auto it = std::find_if(
            outputStracks.begin(), outputStracks.end(),
            [this](const STrack &obj) { return obj.track_id == targetId_; });

        if (it != outputStracks.end()) {
            currentTargetStillExists = true;
            const auto &tlbr = it->tlbr;
            bestBox_ = cv::Rect(tlbr[0], tlbr[1], tlbr[2] - tlbr[0],
                                tlbr[3] - tlbr[1]);
            currentTargetBottomY = tlbr[3];
        }
    }

    for (const STrack &obj : outputStracks) {
        const auto &tlbr = obj.tlbr;
        float h = tlbr[3] - tlbr[1];

        if (h > 400 || obj.score < CONFIDENCE_THRESHOLD)
            continue;

        int classId = obj.classId;

        if ((classId == 2 || classId == 4 || classId == 5) &&
            lanes.size() >= 2) {
            cv::Point bottom_center((tlbr[0] + tlbr[2]) / 2.0f, tlbr[3]);
            std::vector<cv::Point> lane_area = {{lanes[0][0], lanes[0][1]},
                                                {lanes[1][0], lanes[1][1]},
                                                {lanes[1][2], lanes[1][3]},
                                                {lanes[0][2], lanes[0][3]}};

            if (cv::pointPolygonTest(lane_area, bottom_center, false) >= 0) {
                cv::Point center((tlbr[0] + tlbr[2]) / 2.0f,
                                 (tlbr[1] + tlbr[3]) / 2.0f);
                cv::circle(image, center, 5, cv::Scalar(0, 255, 0), -1);

                if (obj.track_id == targetId_) {
                    currentTargetInLane = true;
                    lostTargetCount_ = 0;
                    framesCurrentTargetOutsideLane_ = 0;
                } else if (tlbr[3] > maxBottomY && obj.score > maxConfidence) {
                    bestBoxTmp =
                        cv::Rect(tlbr[0], tlbr[1], tlbr[2] - tlbr[0], h);
                    detectedTargetId = obj.track_id;
                    detectedClassId = obj.classId;
                    maxBottomY = tlbr[3];
                    maxConfidence = obj.score;
                }
            }
        }
    }

    if (targetId_ != -1 && !currentTargetInLane) {
        framesCurrentTargetOutsideLane_++;
    }

    executeEnhancedTargetSwitching(
        detectedTargetId, detectedClassId, bestBoxTmp, currentTargetStillExists,
        currentTargetInLane, maxBottomY, currentTargetBottomY);
}

void EnhancedVideoSubscriber::executeEnhancedTargetSwitching(
    int detectedTargetId, int detectedClassId, cv::Rect bestBoxTmp,
    bool currentTargetStillExists, bool currentTargetInLane, float maxBottomY,
    float currentTargetBottomY) {
    bool shouldSwitchTarget = false;
    std::string switchReason = "";

    if (targetId_ == -1) {
        if (detectedTargetId != -1) {
            shouldSwitchTarget = true;
            switchReason = "No current target";
        }
    } else if (!currentTargetStillExists) {
        lostTargetCount_++;
        if (lostTargetCount_ >= MAX_LOST_FRAMES) {
            if (detectedTargetId != -1) {
                shouldSwitchTarget = true;
                switchReason = "Current target lost";
            } else {
                targetId_ = -1;
                classId_ = -1;
                lostTargetCount_ = 0;
            }
        }
    } else if (detectedTargetId != -1 && detectedTargetId != targetId_) {
        if (framesCurrentTargetOutsideLane_ >= FRAMES_OUTSIDE_LANE) {
            shouldSwitchTarget = true;
            switchReason = "Target outside lane too long";
        } else if (currentTargetInLane &&
                   (maxBottomY - currentTargetBottomY) > DISTANCE_THRESHOLD) {
            shouldSwitchTarget = true;
            switchReason = "Closer target available";
        } else if (!currentTargetInLane) {
            shouldSwitchTarget = true;
            switchReason = "Better target in lane";
        }
    }

    if (shouldSwitchTarget && detectedTargetId != -1) {
        targetId_ = detectedTargetId;
        classId_ = detectedClassId;
        bestBox_ = bestBoxTmp;
        lostTargetCount_ = 0;
        framesCurrentTargetOutsideLane_ = 0;

        if (enable_debug_output_) {
            // ROS_INFO("Target switched to ID %d - Reason: %s", targetId_,
            //          switchReason.c_str());
        }
    }
}

void EnhancedVideoSubscriber::updateSpeedLimits(
    const std::vector<STrack> &outputStracks) {
    bool detectedSpeedLimitThisFrame = false;
    for (const STrack &obj : outputStracks) {
        int classId = obj.classId;
        float conf = obj.score;
        float x1 = obj.tlbr[0];
        float y1 = obj.tlbr[1];
        float x2 = obj.tlbr[2];
        float y2 = obj.tlbr[3];

        int width = static_cast<int>(x2 - x1);
        int height = static_cast<int>(y2 - y1);
        int center_x = static_cast<int>((x1 + x2) / 2.0f);
        int center_y = static_cast<int>((y1 + y2) / 2.0f);

        if (classId >= 12 && classId <= 17 && conf > CONFIDENCE_THRESHOLD &&
            width >= 22) {

            detectedSpeedLimitThisFrame = true;
            int detectedSpeed = (classId - 9) * 10;
            std::string stableSpeedStr =
                speedLimitStabilizer_.update(std::to_string(detectedSpeed));
            int stableSpeed = std::stoi(stableSpeedStr);

            if (stableSpeed != maxSpeed_) {
                maxSpeed_ = stableSpeed;
                if (enable_debug_output_) {
                    printf("📸 Stabilized speed limit updated: %d km/h\n",
                           maxSpeed_);
                }
            }
        }
    }

    if (detectedSpeedLimitThisFrame) {
        noSpeedLimitFrames_ = 0;
    } else {
        noSpeedLimitFrames_++;
    }

    if (noSpeedLimitFrames_ >= 20) {
        speedLimitStabilizer_.clear();
    }
    accSpeed_ = config.speedControl.cruiseSpeedKph;
    // if (maxSpeed_ != -1) {
    //     int targetSpeed =
    //         std::min(maxSpeed_, config.speedControl.cruiseSpeedKph);
    //     if (accSpeed_ < targetSpeed) {
    //         accSpeed_ = std::min(accSpeed_ + 1, targetSpeed);
    //     } else if (accSpeed_ > targetSpeed) {
    //         accSpeed_ = std::max(accSpeed_ - 1, 0);
    //     }
    // } else {
    //     accSpeed_ = config.speedControl.cruiseSpeedKph;
    // }
}

void EnhancedVideoSubscriber::updateFPS() {
    frameCount_++;
    auto now = chrono::steady_clock::now();
    auto elapsed =
        chrono::duration_cast<chrono::seconds>(now - fpsStartTime_).count();

    if (elapsed >= 1) {
        fps_ = frameCount_ / static_cast<double>(elapsed);
        frameCount_ = 0;
        fpsStartTime_ = now;
    }
}