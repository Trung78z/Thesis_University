#ifndef _HEADER_H
#define _HEADER_H
#include <vector>
#include <Detect.h>
int runImages(vector<string> imagePathList, Detect model);
int runVideo(const string path, Detect model);

#endif // _HEADER_H