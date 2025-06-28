#ifndef _HEADER_H
#define _HEADER_H
#include <Detect.h>

#include <vector>
int runImages(vector<string> imagePathList, Detect model);
int runVideo(const string path, Detect model);

#endif  // _HEADER_H