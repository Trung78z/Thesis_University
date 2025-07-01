// config.h - Centralized configuration for camera, detection, and control parameters.
//
// - Camera & Frame Settings: Resolution, FPS, and camera calibration (focal length, object width).
// - ROI: X-axis bounds for focusing detection/tracking.
// - Distance & Speed Estimation: Limits and smoothing for object speed/distance calculations.
// - Adaptive Speed Control: Desired speeds, following distances, and emergency thresholds.
// - Speed Adjustment: How quickly speed can change, and min/max allowed speeds.
// - Object Tracking: Which classes to track, their names, and display colors.
//
// All values are defined as constexpr or const for compile-time safety and easy tuning.
//
// Usage: Include this header wherever these parameters are needed.

#ifndef CONFIG_H
#define CONFIG_H
#include <string>
#include <vector>

namespace Config {

// ========================
// Camera & Frame Settings
// ========================
constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;
constexpr int FPS = 30;
constexpr float focalLength = 1778.0f;   // pixels
constexpr float realObjectWidth = 0.7f;  // meters (average width of a car)

// ==========================
// Region of Interest (ROI)
// ==========================
constexpr float xMin = 290.0f;
constexpr float xMax = 330.0f;

// =============================
// Distance & Speed Estimation
// =============================
constexpr float maxValidSpeedKph = 90.0f;
constexpr float minDistDelta = 0.3f;
constexpr float smoothingFactor = 0.2f;
constexpr float minTimeDelta = 0.15f;
constexpr float minSpeedThreshold = 2.0f;

// ================================
// Adaptive Speed Control Settings
// ================================
constexpr float initialSpeedKph = 60.0f;
constexpr float cruiseSpeedKph = 80.0f;           // Desired cruising speed when no obstacles
constexpr float targetFollowingDistance = 25.0f;  // meters
constexpr float minFollowingDistance = 10.0f;     // meters
constexpr float criticalDistance = 5.0f;          // meters - emergency braking

// ======================================
// Speed Adjustment Parameters (Control)
// ======================================
constexpr float speedUpdateInterval = 0.5f;    // seconds - update interval
constexpr float gentleAdjustment = 2.0f;       // km/h per update (gentle)
constexpr float moderateAdjustment = 5.0f;     // km/h per update (moderate)
constexpr float aggressiveAdjustment = 10.0f;  // km/h per update (urgent)
constexpr float minSpeedKph = 20.0f;
constexpr float maxSpeedKph = 120.0f;

// =======================
// Object Tracking Config
// =======================
const std::vector<int> trackClasses = {0, 1, 2,
                                       3, 5, 7};  // person, bicycle, car, motorcycle, bus, truck

const std::vector<std::string> CLASS_NAMES = {
    "person",          "bicycle",      "car",           "motorcycle",       "bus",
    "truck",           "stop sign",    "other-vehicle", "crosswalk",        "red light",
    "yellow light",    "green light",  "Limit 30km-h",  "Limit 40km-h",     "Limit 50km-h",
    "Limit 60km-h",    "Limit 70km-h", "Limit 80km-h",  "End Limit 60km-h", "End Limit 70km-h",
    "End Limit 80km-h"};

const std::vector<std::vector<unsigned int>> colors = {
    {220, 20, 60},  {119, 172, 48}, {0, 114, 189},   {237, 177, 32},  {126, 47, 142},
    {217, 83, 25},  {255, 0, 0},    {153, 153, 153}, {255, 255, 255}, {255, 0, 0},
    {255, 255, 0},  {0, 255, 0},    {170, 255, 0},   {200, 255, 0},   {255, 255, 0},
    {255, 200, 0},  {255, 170, 0},  {255, 85, 0},    {180, 180, 180}, {140, 140, 140},
    {100, 100, 100}};
}  // namespace Config

#endif  // CONFIG_H