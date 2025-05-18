#include "LaneDetector.hpp"
#include <numeric>

LaneDetector::LaneDetector() {
    // Initialize with default configuration
}

LaneDetector::LaneDetector(const Config& config) : config(config) {
}

void LaneDetector::setConfig(const Config& newConfig) {
    config = newConfig;
}

LaneDetector::Config LaneDetector::getConfig() const {
    return config;
}

cv::Vec4i LaneDetector::makeCoordinates(const cv::Mat& image, const cv::Vec2f& lineParams) const {
    float slope = lineParams[0];
    float intercept = lineParams[1];
    int y1 = static_cast<int>(image.rows * config.yBottomRatio);
    int y2 = static_cast<int>(image.rows * config.yTopRatio);
    int x1 = static_cast<int>((y1 - intercept) / slope);
    int x2 = static_cast<int>((y2 - intercept) / slope);
    return cv::Vec4i(x1, y1, x2, y2);
}

std::vector<cv::Vec4i> LaneDetector::averageSlopeIntercept(const cv::Mat& image, const std::vector<cv::Vec4i>& lines) {
    std::vector<cv::Vec2f> leftFit, rightFit;
    std::vector<cv::Vec4i> averagedLines;

    if (lines.empty()) {
        if (hasPrevLeft && hasPrevRight) {
            return { prevLeft, prevRight };
        }
        return {};
    }

    for (const auto& line : lines) {
        float x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        
        if (x2 == x1) continue;
        float slope = (y2 - y1) / (x2 - x1);
        float intercept = y1 - slope * x1;
        
        if (std::abs(slope) < config.slopeThreshold) {
            continue;
        }
        
        if (slope < 0) {
            leftFit.push_back(cv::Vec2f(slope, intercept));
        } else {
            rightFit.push_back(cv::Vec2f(slope, intercept));
        }
    }

    // Process left lane
    if (!leftFit.empty()) {
        cv::Vec2f leftFitAvg = std::accumulate(leftFit.begin(), leftFit.end(), cv::Vec2f(0, 0)) / static_cast<float>(leftFit.size());
        cv::Vec4i leftLine = makeCoordinates(image, leftFitAvg);
        
        if (hasPrevLeft) {
            leftLine = cv::Vec4i(
                static_cast<int>((1 - smoothingFactor) * leftLine[0] + smoothingFactor * prevLeft[0]),
                static_cast<int>((1 - smoothingFactor) * leftLine[1] + smoothingFactor * prevLeft[1]),
                static_cast<int>((1 - smoothingFactor) * leftLine[2] + smoothingFactor * prevLeft[2]),
                static_cast<int>((1 - smoothingFactor) * leftLine[3] + smoothingFactor * prevLeft[3])
            );
        }
        
        averagedLines.push_back(leftLine);
        prevLeft = leftLine;
        hasPrevLeft = true;
    } else if (hasPrevLeft) {
        averagedLines.push_back(prevLeft);
    }

    // Process right lane
    if (!rightFit.empty()) {
        cv::Vec2f rightFitAvg = std::accumulate(rightFit.begin(), rightFit.end(), cv::Vec2f(0, 0)) / static_cast<float>(rightFit.size());
        cv::Vec4i rightLine = makeCoordinates(image, rightFitAvg);
        
        if (hasPrevRight) {
            rightLine = cv::Vec4i(
                static_cast<int>((1 - smoothingFactor) * rightLine[0] + smoothingFactor * prevRight[0]),
                static_cast<int>((1 - smoothingFactor) * rightLine[1] + smoothingFactor * prevRight[1]),
                static_cast<int>((1 - smoothingFactor) * rightLine[2] + smoothingFactor * prevRight[2]),
                static_cast<int>((1 - smoothingFactor) * rightLine[3] + smoothingFactor * prevRight[3])
            );
        }
        
        averagedLines.push_back(rightLine);
        prevRight = rightLine;
        hasPrevRight = true;
    } else if (hasPrevRight) {
        averagedLines.push_back(prevRight);
    }

    return averagedLines;
}

cv::Mat LaneDetector::canny(const cv::Mat& image) const {
    cv::Mat gray, blur, edges;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, config.blurKernel, 0);
    cv::Canny(blur, edges, config.cannyThresholds[0], config.cannyThresholds[1]);
    return edges;
}

cv::Mat LaneDetector::regionOfInterest(const cv::Mat& image) const {
    int height = image.rows;
    
    std::vector<cv::Point> vertices = {
        cv::Point(config.x1, height),
        cv::Point(config.x2, height),
        cv::Point(config.x4, config.y),
        cv::Point(config.x3, config.y)
    };
    
    cv::Mat mask = cv::Mat::zeros(image.size(), image.type());
    std::vector<std::vector<cv::Point>> pts = { vertices };
    cv::fillPoly(mask, pts, cv::Scalar(255));
    cv::Mat masked;
    cv::bitwise_and(image, mask, masked);
    return masked;
}

cv::Mat LaneDetector::displayLines(const cv::Mat& image, const std::vector<cv::Vec4i>& lines) const {
    cv::Mat lineImage = cv::Mat::zeros(image.size(), image.type());
    if (!lines.empty()) {
        for (const auto& line : lines) {
            cv::line(lineImage, 
                    cv::Point(line[0], line[1]), 
                    cv::Point(line[2], line[3]), 
                    config.lineColor, 
                    config.lineThickness);
        }
    }
    return lineImage;
}

std::pair<cv::Mat, std::vector<cv::Vec4i>> LaneDetector::processImage(const cv::Mat& image) {
    cv::Mat laneImage = image.clone();
    
    if (image.channels() == 1) {
        cv::cvtColor(image, laneImage, cv::COLOR_GRAY2BGR);
    }
    
    cv::Mat cannyImage = canny(laneImage);
    cv::Mat croppedImage = regionOfInterest(cannyImage);
    
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(croppedImage, lines, 
                   config.houghRho, config.houghTheta, 
                   config.houghThreshold,
                   config.minLineLength, config.maxLineGap);
    
    std::vector<cv::Vec4i> averagedLines = averageSlopeIntercept(laneImage, lines);
    cv::Mat lineImage = displayLines(laneImage, averagedLines);
    cv::Mat comboImage;
    cv::addWeighted(laneImage, 0.8, lineImage, 1, 1, comboImage);
    
    return { comboImage, averagedLines };
}

void LaneDetector::visualize(const cv::Mat& image) {
    auto result = processImage(image);
    cv::imshow("Lane Detection", result.first);
    cv::waitKey(0);
}