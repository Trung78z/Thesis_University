#ifndef ENHANCED_VIDEO_SUBSCRIBER_H
#define ENHANCED_VIDEO_SUBSCRIBER_H

#include <chrono>

#include <deque>

#include <map>
#include <opencv2/opencv.hpp>

#include "EgoVehicle.h"
#include "HUDRenderer.h"
#include "TrafficSignStabilizer.hpp"
#include "VideoRecorder.h"
#include <BYTETracker.h>
#include <Detect.h>
#include <LaneDetector.h>
#include <utils.hpp>
#include <Logger.h>
class EnhancedVideoSubscriber {
private:
    std::string model_path_;
    std::string output_dir_;
    bool enable_debug_output_;
    bool enable_data_logging_;

    Detect model_;
    LaneDetector laneDetector_;
    BYTETracker tracker_;
    int frameCount_;
    std::chrono::steady_clock::time_point fpsStartTime_;
    double fps_;

    int maxSpeed_;
    int accSpeed_;
    float currentEgoSpeed_;
    double lastSpeedUpdateTime_;
    std::deque<float> speedChangeHistory_;
    std::deque<float> distanceHistory_;

    std::map<int, std::deque<float>> objectBuffers_;
    std::map<int, float> prevDistances_;
    std::map<int, double> prevTimes_;
    std::map<int, float> smoothedSpeeds_;

    int targetId_;
    int classId_;
    cv::Rect bestBox_;
    int lostTargetCount_;
    static constexpr int MAX_LOST_FRAMES = 8;
    static constexpr float DISTANCE_THRESHOLD = 40.0f;
    static constexpr int FRAMES_OUTSIDE_LANE = 8;
    static constexpr float CONFIDENCE_THRESHOLD = 0.8f;
    int framesCurrentTargetOutsideLane_;
    int noSpeedLimitFrames_;

    VideoRecorder videoRecorder_;
    HUDRenderer hudRenderer_;

    std::deque<double> processing_times_;
    int total_detections_;

    bool emergency_stop_;
    double last_detection_time_;
    static constexpr double DETECTION_TIMEOUT = 2.0;

    EgoVehicle egoVehicle_;
    TrafficSignStabilizer speedLimitStabilizer_;

    void initializeEnhancedFeatures();
    void createDirectory(const std::string &path);
    std::string getModelPath();
    void publishEnhancedData(float ego_speed, const std::string &action_str,
                             float throttle_cmd, float brake_cmd);
    void checkSafetyConditions();
    void processEnhancedFrame(cv::Mat &image);
    void
    performEnhancedTargetSelection(const std::vector<STrack> &outputStracks,
                                   const std::vector<cv::Vec4i> &lanes,
                                   cv::Mat &image);
    void executeEnhancedTargetSwitching(
        int detectedTargetId, int detectedClassId, cv::Rect bestBoxTmp,
        bool currentTargetStillExists, bool currentTargetInLane,
        float maxBottomY, float currentTargetBottomY);
    void updateSpeedLimits(const std::vector<STrack> &outputStracks);
    void updateFPS();

public:
    EnhancedVideoSubscriber(const string &model_path_);
    ~EnhancedVideoSubscriber();
    void imageCallback(const cv::Mat &msg);
};

#endif // ENHANCED_VIDEO_SUBSCRIBER_H