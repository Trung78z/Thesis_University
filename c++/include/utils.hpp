#ifndef _UTILS_H
#define _UTILS_H

#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
using namespace std;
bool checkVideo(const string &path);
bool checkImages(const string &path, vector<string> &imagePathList);
#endif