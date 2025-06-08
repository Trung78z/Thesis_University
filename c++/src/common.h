#ifndef COMMON_H
#define COMMON_H
#include <iostream>
#include <vector>

const std::vector<std::string> CLASS_NAMES = {
    "person", "bike", "car", "motorcycle", "bus", "truck", 
    "traffic-light", "stop-sign", "vehicle", 
    "limit", "limit-50", "limit-60", "limit-70", "limit-80", "limit-90"};

const std::vector<std::vector<unsigned int>> COLORS = {
    // Objects - Distinct colors
    {220, 20, 60},    // person (red)
    {119, 172, 48},   // bike (green)
    {0, 114, 189},    // car (blue)
    {237, 177, 32},   // motorcycle (yellow)
    {126, 47, 142},   // bus (purple)
    {217, 83, 25},    // truck (orange)
    {162, 20, 47},    // traffic-light (dark red)
    {76, 76, 76},     // stop-sign (dark gray)
    {153, 153, 153},  // vehicle (gray)
    
    // Speed limits - Gradient from green to red
    {0, 255, 0},      // limit (bright green)
    {85, 255, 0},     // limit-50
    {170, 255, 0},    // limit-60
    {255, 255, 0},    // limit-70 (yellow)
    {255, 170, 0},    // limit-80 (orange)
    {255, 85, 0}      // limit-90 (red-orange)
};

#endif // COMMON_H