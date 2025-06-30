// config.h
#pragma once
#include <string>
#include <vector>

namespace Config {
constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;
constexpr int FPS = 30;
constexpr float FOCAL_LENGTH = 1778.0f;

const std::vector<int> trackClasses = {0, 1, 2, 3, 5, 7};

const std::vector<std::string> CLASS_NAMES = {
    "person",          "bicycle",      "car",           "motorcycle",       "bus",
    "truck",           "stop sign",    "other-vehicle", "crosswalk",        "red light",
    "yellow light",    "green light",  "Limit 30km-h",  "Limit 40km-h",     "Limit 50km-h",
    "Limit 60km-h",    "Limit 70km-h", "Limit 80km-h",  "End Limit 60km-h", "End Limit 70km-h",
    "End Limit 80km-h"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {220, 20, 60},  {119, 172, 48}, {0, 114, 189},   {237, 177, 32},  {126, 47, 142},
    {217, 83, 25},  {255, 0, 0},    {153, 153, 153}, {255, 255, 255}, {255, 0, 0},
    {255, 255, 0},  {0, 255, 0},    {170, 255, 0},   {200, 255, 0},   {255, 255, 0},
    {255, 200, 0},  {255, 170, 0},  {255, 85, 0},    {180, 180, 180}, {140, 140, 140},
    {100, 100, 100}};
}  // namespace Config
