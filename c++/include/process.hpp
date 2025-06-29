#ifndef _PROCESS_HPP_
#define _PROCESS_HPP_

#include <BYTETracker.h>
#include <LaneDetector.h>

#include <cxxopts.hpp>
#include <header.hpp>
#include <iostream>
#include <string>
#include <utils.hpp>
#include <vector>

#include "Detect.h"
#include "common/config.h"

bool isTrackingClass(int class_id);
int runVideo(const string path, Detect model);
int runImages(vector<string> imagePathList, Detect model);

#endif