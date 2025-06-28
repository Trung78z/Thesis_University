#ifndef _UTILS_H
#define _UTILS_H

#include <iostream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
using namespace std;
bool checkVideo(const string& path);
bool checkImages(const string &path, vector<string> &imagePathList);
#endif