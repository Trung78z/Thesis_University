#include <EgoVehicle.h>
#include <process.h>

#include <numeric>

int runImages(const vector<string> imagePathList, Detect &model) {
    // Path to folder containing images
    int fps = 30;
    BYTETracker tracker(fps, 30);
    for (const auto &imagePath : imagePathList) {
        // Open image
        Mat image = imread(imagePath);
        if (image.empty()) {
            cerr << "Error reading image: " << imagePath << endl;
            continue;
        }

        vector<Detection> res;
        model.preprocess(image);

        auto start = std::chrono::system_clock::now();
        model.infer();
        auto end = std::chrono::system_clock::now();

        model.postprocess(image, res);
        std::vector<Object> objects = filterDetections(res);
        std::vector<STrack> output_stracks = tracker.update(objects);
        model.draw(image, output_stracks);
        auto tc =
            (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
            1000.;
        printf("cost %2.4lf ms\n", tc);

        imshow("Result", image);

        waitKey(0);
    }
    return 0;
}

int runVideo(const std::string &path, Detect &model) {
    cout << "Opening video: " << path << endl;
    // Example GStreamer pipeline for Jetson (commented out)
    // cv::VideoCapture('nvarguscamerasrc !
    // video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv !
    // video/x-raw,format=BGRx ! videoconvert ! appsink', cv::CAP_GSTREAMER);
    // std::string capture_pipeline =
    //     "nvarguscamerasrc ! "
    //     "video/x-raw(memory:NVMM), width=" + std::to_string(width) +
    //     ", height=" + std::to_string(height) +
    //     ", format=NV12, framerate=" + std::to_string(fps) + "/1 ! "
    //     "nvvidconv flip-method=0 ! "
    //     "video/x-raw, format=BGRx ! "
    //     "videoconvert ! "
    //     "video/x-raw, format=BGR ! "
    //     "appsink drop=true";
    // cv::VideoCapture cap(capture_pipeline, cv::CAP_GSTREAMER);
    cv::VideoCapture cap(path);

    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video file!" << endl;
        return 0;
    }
    // Get frame width, height, and fps
    double fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    BYTETracker tracker(fps, 30);
    int frameCount = 0;
    auto fpsStartTime = std::chrono::steady_clock::now();
    int maxSpeed = -1;     // km/h
    int accMaxSpeed = 80;  // km/h
    int accSpeed = 60;     // km/h
    int total_ms = 0;
    LaneDetector laneDetector;
    while (cap.isOpened()) {
        auto now = std::chrono::steady_clock::now();
        auto start = std::chrono::system_clock::now();
        cv::Mat image;
        cap >> image;

        if (image.empty()) {
            break;
        }
        // Resize the image to fit the window
        // cv::resize(image, image, cv::Size(width, height));
        vector<Detection> res;

        model.preprocess(image);

        model.infer();

        model.postprocess(image, res);

        std::vector<Object> objects = filterDetections(res);

        std::vector<cv::Vec4i> lanes = laneDetector.detectLanes(image);
        // Tracking
        std::vector<STrack> output_stracks = tracker.update(objects);

        auto end = std::chrono::system_clock::now();
        total_ms = total_ms + getTotalMilliseconds(start, end);

        model.draw(image, output_stracks);
        // Log detected objects
        for (const auto &obj : res) {
            auto box = obj.bbox;
            auto classId = obj.classId;
            auto conf = obj.conf;

            cv::Point bottom_center(box.x + box.width / 2, box.y + box.height);
            if (lanes.size() >= 2) {
                // Get 2 lanes (left - right)
                cv::Vec4i l0 = lanes[0];
                cv::Vec4i l1 = lanes[1];

                // Calculate the middle point of the lane to draw a line
                cv::Point laneTop((l0[0] + l1[0]) / 2, (l0[1] + l1[1]) / 2);
                cv::Point laneBottom((l0[2] + l1[2]) / 2, (l0[3] + l1[3]) / 2);

                // Create a polygon for the lane area in order: top-left → top-right → bottom-right
                // → bottom-left
                std::vector<cv::Point> lane_area = {
                    cv::Point(l0[0], l0[1]),  // top-left
                    cv::Point(l1[0], l1[1]),  // top-right
                    cv::Point(l1[2], l1[3]),  // bottom-right
                    cv::Point(l0[2], l0[3])   // bottom-left
                };

                if (obj.classId == 2 || obj.classId == 4 ||
                    obj.classId == 5)  // only consider car/bus/truck
                {
                    if (cv::pointPolygonTest(lane_area, bottom_center, false) >=
                        0)  // vehicle is in the lane
                    {
                        // Draw a dot at the center of the bbox
                        cv::Point mid(box.x + box.width / 2,
                                      box.y + box.height / 2);                 // bbox center
                        cv::circle(image, mid, 5, cv::Scalar(0, 255, 0), -1);  // green dot
                    }
                }
            }

            if (classId >= 12 && classId <= 17 && conf > 0.6) {
                int newSpeed = (classId - 9) * 10;  // classId 12 -> 30km/h, 13 -> 40, etc.
                maxSpeed = newSpeed;
            }
        }

        laneDetector.drawLanes(image, lanes);
        // fps calculation
        frameCount++;

        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - fpsStartTime).count();
        if (elapsed >= 1) {
            fps = frameCount / static_cast<double>(elapsed);
            frameCount = 0;
            fpsStartTime = now;
        }
        // Draw max speed
        if (maxSpeed != -1) {
            cv::putText(image, cv::format("Max Speed: %dKm/h", maxSpeed), cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        } else {
            cv::putText(image, "No Speed Limit Detected", cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        }

        // Draw Cruise Control
        cv::putText(image, cv::format("Max Cruise Control: %dKm/h", accMaxSpeed), cv::Point(10, 90),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        // Draw current speed
        cv::putText(image, cv::format("Control speed: %dKm/h", accSpeed), cv::Point(10, 120),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        if (maxSpeed != -1) {
            if (accSpeed < maxSpeed && accSpeed < accMaxSpeed) {
                accSpeed += 1;  // Increase speed by 1 km/h
            } else if (accSpeed > maxSpeed && accSpeed > 0) {
                accSpeed -= 1;  // Decrease speed by 1 km/h
            }
        } else {
            accSpeed = accMaxSpeed;  // Reset to max speed if no speed limit detected
        }
        // Draw fps
        cv::putText(image, cv::format("fps: %.2f", fps), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Result", image);
        if (cv::waitKey(1) == 'q') {  // Press 'q' to exit
            break;
        }
    }

    // Release resources
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

std::vector<Object> filterDetections(const std::vector<Detection> &res) {
    std::vector<Object> objects;
    for (const auto &obj : res) {
        if (isTrackingClass(obj.classId)) {
            objects.push_back({obj.bbox, obj.classId, obj.conf});
        }
    }
    return objects;
}

void selectTarget(const std::vector<STrack> &tracks, float xMin, float xMax, int &targetId,
                  cv::Rect &bestBox, float &maxHeight) {
    for (const auto &track : tracks) {
        if (!track.is_activated) continue;
        auto &tlbr = track.tlbr;
        float x1 = tlbr[0], y1 = tlbr[1], x2 = tlbr[2], y2 = tlbr[3];
        float xCenter = (x1 + x2) / 2.0f;
        float height = y2 - y1;

        if (xMin <= xCenter && xCenter <= xMax && height > maxHeight) {
            maxHeight = height;
            targetId = track.track_id;
            bestBox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
        }
    }
}

void updateSpeedControl(double timeStart, int targetId, const cv::Rect &bestBox,
                        float &currentEgoSpeed, double &lastSpeedUpdateTime,
                        std::map<int, std::deque<float>> &objectBuffers,
                        std::map<int, float> &prevDistances, std::map<int, double> &prevTimes,
                        std::map<int, float> &smoothedSpeeds, std::deque<float> &speedChangeHistory,
                        float &avgDistance, float &frontSpeed, std::string &action,
                        cv::Scalar &actionColor) {
    if (targetId != -1 && bestBox.height > 0) {
        float h = bestBox.height;
        float distance = (realObjectWidth * focalLength) / h;

        // Initialize if new
        if (objectBuffers.find(targetId) == objectBuffers.end()) {
            objectBuffers[targetId] = std::deque<float>();
            prevDistances[targetId] = distance;
            prevTimes[targetId] = timeStart;
            smoothedSpeeds[targetId] = 0.0f;
        }

        // Push to buffer
        auto &buf = objectBuffers[targetId];
        buf.push_back(distance);
        if (buf.size() > 5) buf.pop_front();

        // ✅ Always update avgDistance
        if (buf.size() >= 3) {
            std::deque<float> sortedBuf = buf;
            std::sort(sortedBuf.begin(), sortedBuf.end());
            avgDistance = sortedBuf[sortedBuf.size() / 2];  // median
        } else {
            avgDistance = std::accumulate(buf.begin(), buf.end(), 0.0f) / buf.size();  // mean
        }

        // ✅ Always update smoothed speed
        double dt = timeStart - prevTimes[targetId];
        if (dt >= minTimeDelta) {
            float dDist = prevDistances[targetId] - avgDistance;
            if (std::abs(dDist) >= minDistDelta) {
                float speed = (dDist / dt) * 3.6f;
                smoothedSpeeds[targetId] =
                    smoothingFactor * speed + (1 - smoothingFactor) * smoothedSpeeds[targetId];
                prevDistances[targetId] = avgDistance;
                prevTimes[targetId] = timeStart;
            }
        }

        // ✅ Always update front speed every frame
        float relativeSpeed = smoothedSpeeds[targetId];
        frontSpeed = currentEgoSpeed - relativeSpeed;

        // Speed control logic (less frequent)
        if (timeStart - lastSpeedUpdateTime >= speedUpdateInterval) {
            auto [state, urgency] = getDrivingState(avgDistance, frontSpeed, currentEgoSpeed);
            float targetSpeed =
                calculateTargetSpeed(avgDistance, frontSpeed, currentEgoSpeed, state, urgency);
            float oldSpeed = currentEgoSpeed;

            currentEgoSpeed = updateEgoSpeedSmooth(currentEgoSpeed, targetSpeed, urgency,
                                                   timeStart - lastSpeedUpdateTime);
            float speedDelta = currentEgoSpeed - oldSpeed;

            speedChangeHistory.push_back(speedDelta);
            if (speedChangeHistory.size() > 10) speedChangeHistory.pop_front();

            lastSpeedUpdateTime = timeStart;
            getActionAndColor(state, speedDelta, action, actionColor);

            std::cout << "[+] ID " << targetId << " | Dist: " << std::fixed << std::setprecision(1)
                      << avgDistance << "m | Front: " << frontSpeed
                      << " km/h | Ego: " << currentEgoSpeed << " km/h | State: " << state
                      << " | Action: " << action << std::endl;
        }
    } else {
        // No target: cruise mode
        if (timeStart - lastSpeedUpdateTime >= speedUpdateInterval) {
            if (std::abs(currentEgoSpeed - cruiseSpeedKph) > 1) {
                currentEgoSpeed +=
                    (currentEgoSpeed < cruiseSpeedKph) ? gentleAdjustment : -gentleAdjustment;
                currentEgoSpeed = std::clamp(currentEgoSpeed, 0.0f, maxValidSpeedKph);
            }
            lastSpeedUpdateTime = timeStart;
        }

        // ❗ Reset or mark frontSpeed and avgDistance as unavailable
        frontSpeed = 0.0f;
        avgDistance = -1.0f;
        action = "CRUISE";
        actionColor = cv::Scalar(200, 200, 200);
    }
}
