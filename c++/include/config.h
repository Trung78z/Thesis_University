#ifndef CONFIG_H
#define CONFIG_H

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace Config {

// Structs for organization
struct CameraSettings {
    int width;
    int height;
    int fps;
    float focalLength;
    float realObjectWidth;
};

struct ROI {
    float xMin;
    float xMax;
};

struct Estimation {
    float maxValidSpeedKph;
    float minDistDelta;
    float smoothingFactor;
    float minTimeDelta;
    float minSpeedThreshold;
};

struct AdaptiveSpeed {
    float initial;
    float cruise;
    float followDistance;
    float minFollowDistance;
    float criticalDistance;
};

struct Adjustment {
    float interval;
    float gentle;
    float moderate;
    float aggressive;
    float min;
    float max;
};

struct Tracking {
    std::vector<int> trackClasses;
    std::vector<std::string> classNames;
    std::vector<std::vector<unsigned int>> colors;
};

struct HUDColors {
    cv::Scalar white, red, yellow, orange, green, gray;
};

// Config variables
extern CameraSettings camera;
extern ROI roi;
extern Estimation estimation;
extern AdaptiveSpeed adaptiveSpeed;
extern Adjustment adjustment;
extern Tracking tracking;
extern HUDColors hudColors;

// Loader
bool LoadConfig(const std::string& path);

}  // namespace Config

#endif  // CONFIG_H
