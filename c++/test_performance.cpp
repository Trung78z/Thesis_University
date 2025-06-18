#include "include/lanevision/LaneDetector.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>

int main() {
    // Create lane detector
    LaneDetector detector;
    
    // Create a test image (simulate road scene)
    cv::Mat test_image = cv::Mat::zeros(480, 640, CV_8UC3);
    
    // Draw some simulated lane lines
    cv::line(test_image, cv::Point(200, 400), cv::Point(250, 300), cv::Scalar(255, 255, 255), 3);
    cv::line(test_image, cv::Point(440, 400), cv::Point(390, 300), cv::Scalar(255, 255, 255), 3);
    
    // Add some noise
    cv::randn(test_image, cv::Scalar(50, 50, 50), cv::Scalar(30, 30, 30));
    
    std::cout << "Lane Detection Performance Test" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Warm up
    for (int i = 0; i < 10; ++i) {
        detector.detectLanes(test_image);
    }
    
    // Performance test
    const int num_iterations = 100;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        auto lanes = detector.detectLanes(test_image);
        
        // Optional: draw lanes to test drawing performance
        if (i % 10 == 0) {  // Only draw every 10th frame to reduce overhead
            cv::Mat result = test_image.clone();
            detector.drawLanes(result, lanes);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    double avg_time_ms = duration.count() / (1000.0 * num_iterations);
    double fps = 1000.0 / avg_time_ms;
    
    std::cout << "Test Results:" << std::endl;
    std::cout << "  Total iterations: " << num_iterations << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) 
              << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "  Average time per frame: " << std::fixed << std::setprecision(2) 
              << avg_time_ms << " ms" << std::endl;
    std::cout << "  Estimated FPS: " << std::fixed << std::setprecision(1) << fps << std::endl;
    
    // Test with actual detection
    auto lanes = detector.detectLanes(test_image);
    std::cout << "  Detected lanes: " << lanes.size() << std::endl;
    
    // Show result
    cv::Mat result = test_image.clone();
    detector.drawLanes(result, lanes);
    
    cv::imshow("Original", test_image);
    cv::imshow("Lane Detection Result", result);
    cv::waitKey(0);
    
    return 0;
} 