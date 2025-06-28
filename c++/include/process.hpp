#ifndef _PROCESS_HPP_
#define _PROCESS_HPP_

#include <BYTETracker.h>
#include <lanevision/LaneDetector.h>

#include <cxxopts.hpp>
#include <header.hpp>
#include <iostream>
#include <string>
#include <utils.hpp>
#include <vector>

#include "Detect.h"
#include "common/common.h"

const std::vector<int> trackClasses{0, 1, 2,
                                    3, 5, 7};  // person, bicycle, car, motorcycle, bus, truck

bool isTrackingClass(int class_id);
int runVideo(const string path, Detect model);
int runImages(vector<string> imagePathList, Detect model);

#endif