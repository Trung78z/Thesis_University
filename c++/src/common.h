#ifndef COMMON_H
#define COMMON_H
#include <iostream>
#include <vector>

const std::vector<std::string> CLASS_NAMES = {
    "person",                  // 0
    "bicycle",                 // 1
    "car",                     // 2
    "motorcycle",              // 3
    "bus",                     // 4
    "truck",                   // 5
    "other-vehicle",           // 6
    "traffic light",           // 7
    "stop sign",               // 8
    "Speed limit",             // 9
    "Speed limit 20km-h",      // 10
    "Speed limit 30km-h",      // 11
    "Speed limit 40km-h",      // 12
    "Speed limit 50km-h",      // 13
    "Speed limit 60km-h",      // 14
    "Speed limit 70km-h",      // 15
    "Speed limit 80km-h",      // 16
    "Speed limit 100km-h",     // 17
    "Speed limit 120km-h",     // 18
    "End of speed limit 80km-h"// 19
};

const std::vector<std::vector<unsigned int>> COLORS = {
    {220, 20, 60},    // person (red)
    {119, 172, 48},   // bicycle (green)
    {0, 114, 189},    // car (blue)
    {237, 177, 32},   // motorcycle (yellow)
    {126, 47, 142},   // bus (purple)
    {217, 83, 25},    // truck (orange)
    {153, 153, 153},  // other-vehicle (gray)
    {162, 20, 47},    // traffic light (dark red)
    {76, 76, 76},     // stop sign (dark gray)
    {0, 255, 0},      // Speed limit (bright green)
    {85, 255, 0},     // Speed limit 20km-h
    {170, 255, 0},    // Speed limit 30km-h
    {200, 255, 0},    // Speed limit 40km-h
    {255, 255, 0},    // Speed limit 50km-h (yellow)
    {255, 200, 0},    // Speed limit 60km-h
    {255, 170, 0},    // Speed limit 70km-h (orange)
    {255, 85, 0},     // Speed limit 80km-h (red-orange)
    {255, 0, 0},      // Speed limit 100km-h (red)
    {255, 0, 85},     // Speed limit 120km-h (pinkish red)
    {128, 128, 128}   // End of speed limit 80km-h (neutral gray)
};

#endif // COMMON_H