#ifndef LANE_DETECTOR_HPP
#define LANE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class LaneDetector {
public:
    struct Config {
        cv::Vec2i cannyThresholds = cv::Vec2i(50, 150);
        cv::Size blurKernel = cv::Size(5, 5);
        double houghRho = 2;
        double houghTheta = CV_PI / 180;
        int houghThreshold = 100;
        int minLineLength = 40;
        int maxLineGap = 5;
        std::vector<cv::Point> roiVertices;
        cv::Scalar lineColor = cv::Scalar(255, 150, 0);
        int lineThickness = 10;
        float yBottomRatio = 0.95f;
        float yTopRatio = 0.45f;
        float slopeThreshold = 0.5f;
        int x1 = 200, x2 = 1100, x3 = 470, x4 = 730, y = 400;
    };

    LaneDetector();
    explicit LaneDetector(const Config& config);

    void setConfig(const Config& newConfig);
    Config getConfig() const;

    std::pair<cv::Mat, std::vector<cv::Vec4i>> processImage(const cv::Mat& image);
    void visualize(const cv::Mat& image);

private:
    cv::Vec4i makeCoordinates(const cv::Mat& image, const cv::Vec2f& lineParams) const;
    std::vector<cv::Vec4i> averageSlopeIntercept(const cv::Mat& image, const std::vector<cv::Vec4i>& lines);
    cv::Mat canny(const cv::Mat& image) const;
    cv::Mat regionOfInterest(const cv::Mat& image) const;
    cv::Mat displayLines(const cv::Mat& image, const std::vector<cv::Vec4i>& lines) const;

    Config config;
    cv::Vec4i prevLeft;
    cv::Vec4i prevRight;
    bool hasPrevLeft = false;
    bool hasPrevRight = false;
    float smoothingFactor = 0.2f;
};

#endif // LANE_DETECTOR_HPP