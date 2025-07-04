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
using namespace Config;
int runVideo(const std::string &path, Detect &model);
int runImages(const vector<string> imagePathList, Detect &model);

void selectTarget(const std::vector<STrack> &tracks, float xMin, float xMax,
                  int &targetId, cv::Rect &bestBox, float &maxHeight);
std::vector<Object> filterDetections(const std::vector<Detection> &res);

void drawHUD(cv::Mat &image, float currentEgoSpeed, int accSpeed, int maxSpeed,
             float frontSpeed, float avgDistance, const std::string &action,
             const cv::Scalar &actionColor, double fps, int targetId);

#endif