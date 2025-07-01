#ifndef _PROCESS_HPP_
#define _PROCESS_HPP_

#include <BYTETracker.h>
#include <LaneDetector.h>

#include <cxxopts.hpp>
#include <iostream>
#include <string>
#include <utils.hpp>
#include <vector>

#include "Detect.h"
#include "config.h"

bool isTrackingClass(int classId);
int runVideo(const string path, Detect model);
int runImages(vector<string> imagePathList, Detect model);

#endif