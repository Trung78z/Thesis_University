// FrontDistanceEstimator.h
#pragma once
#include <iostream>
class FrontDistanceEstimator {
public:
    FrontDistanceEstimator();
    double estimate(double pixelDistance, double focalLength, double realObjectWidth);
};
