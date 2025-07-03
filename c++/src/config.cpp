#include "config.h"

#include <fstream>
#include <iostream>

namespace Config {

// Default fallback (in case JSON is missing)
CameraSettings camera = {1280, 720, 30, 1778.0f, 0.7f};
ROI roi = {290.0f, 330.0f};
Estimation estimation = {90.0f, 0.3f, 0.2f, 0.15f, 2.0f};
AdaptiveSpeed adaptiveSpeed = {60.0f, 80.0f, 25.0f, 10.0f, 5.0f};
Adjustment adjustment = {0.5f, 1.0f, 5.0f, 10.0f, 20.0f, 120.0f};
Tracking tracking = {};
HUDColors hudColors = {
    {255, 255, 255},  // white
    {68, 68, 255},    // red
    {0, 255, 255},    // yellow
    {0, 165, 255},    // orange
    {102, 255, 102},  // green
    {180, 180, 180}   // gray
};

bool LoadConfig(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Config] Cannot open config file: " << path << "\n";
        return false;
    }

    nlohmann::json j;
    file >> j;

    auto c = j["camera"];
    camera = {c["width"], c["height"], c["fps"], c["focal_length"], c["real_object_width"]};

    auto r = j["roi"];
    roi = {r["x_min"], r["x_max"]};

    auto e = j["estimation"];
    estimation = {e["max_valid_speed_kph"], e["min_dist_delta"], e["smoothing_factor"],
                  e["min_time_delta"], e["min_speed_threshold"]};

    auto a = j["adaptive_speed"];
    adaptiveSpeed = {a["initial"], a["cruise"], a["follow_distance"], a["min_follow_distance"],
                     a["critical_distance"]};

    auto ad = j["adjustment"];
    adjustment = {ad["interval"],   ad["gentle"], ad["moderate"],
                  ad["aggressive"], ad["min"],    ad["max"]};

    auto t = j["tracking"];
    tracking.trackClasses = t["track_classes"].get<std::vector<int>>();
    tracking.classNames = t["class_names"].get<std::vector<std::string>>();
    tracking.colors = t["colors"].get<std::vector<std::vector<unsigned int>>>();

    auto h = j["hud_colors"];
    hudColors = {{h["white"][0], h["white"][1], h["white"][2]},
                 {h["red"][0], h["red"][1], h["red"][2]},
                 {h["yellow"][0], h["yellow"][1], h["yellow"][2]},
                 {h["orange"][0], h["orange"][1], h["orange"][2]},
                 {h["green"][0], h["green"][1], h["green"][2]},
                 {h["gray"][0], h["gray"][1], h["gray"][2]}};

    std::cout << "[Config] Loaded config from: " << path << "\n";
    return true;
}

}  // namespace Config
