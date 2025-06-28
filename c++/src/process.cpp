#include <process.hpp>

bool isTrackingClass(int class_id) {
    return true;
    // for (auto &c : trackClasses) {
    //   if (class_id == c) return true;
    // }
    // return false;
}

int runImages(vector<string> imagePathList, Detect model) {
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
        std::vector<Object> objects;
        for (const auto &obj : res) {
            auto box = obj.bbox;
            auto class_id = obj.class_id;
            auto conf = obj.conf;

            if (isTrackingClass(class_id)) {
                Object obj{box, class_id, conf};
                objects.push_back(obj);
            }
        }
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

int runVideo(const string path, Detect model) {
    cout << "Opening video: " << path << endl;
    // Example GStreamer pipeline for Jetson (commented out)
    // cv::VideoCapture('nvarguscamerasrc !
    // video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv !
    // video/x-raw,format=BGRx ! videoconvert ! appsink', cv::CAP_GSTREAMER);
    // std::string capture_pipeline =
    //     "nvarguscamerasrc ! "
    //     "video/x-raw(memory:NVMM), width=" + std::to_string(WIDTH) +
    //     ", height=" + std::to_string(HEIGHT) +
    //     ", format=NV12, framerate=" + std::to_string(FPS) + "/1 ! "
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
    // Get frame width, height, and FPS
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
        // cv::resize(image, image, cv::Size(WIDTH, HEIGHT));
        vector<Detection> res;

        model.preprocess(image);

        model.infer();

        model.postprocess(image, res);

        std::vector<Object> objects;
        for (const auto &obj : res) {
            auto box = obj.bbox;
            auto class_id = obj.class_id;
            auto conf = obj.conf;

            if (isTrackingClass(class_id)) {
                Object obj{box, class_id, conf};
                objects.push_back(obj);
            }
        }

        std::vector<cv::Vec4i> lanes = laneDetector.detectLanes(image);
        // Tracking
        std::vector<STrack> output_stracks = tracker.update(objects);

        auto end = std::chrono::system_clock::now();
        total_ms =
            total_ms + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // for (int i = 0; i < output_stracks.size(); i++) {
        //   std::ostringstream label_ss;
        //   label_ss << CLASS_NAMES[output_stracks[i].class_id] << " " << std::fixed
        //            << std::setprecision(2) << output_stracks[i].score;
        //   std::string label = label_ss.str();
        //   // std::cout << "Track ID: " << output_stracks[i].track_id
        //   //           << ", Class ID: " << output_stracks[i].class_id
        //   //           << ", Score: " << output_stracks[i].score << ", TLWH: " <<
        //   //           output_stracks[i].tlwh[0]
        //   //           << ", " << output_stracks[i].tlwh[1] << ", " << output_stracks[i].tlwh[2]
        //   <<
        //   ", "
        //   //           << output_stracks[i].tlwh[3] << std::endl;
        //   std::vector<float> tlwh = output_stracks[i].tlwh;
        //   // bool vertical = tlwh[2] / tlwh[3] > 1.6;
        //   // if (tlwh[2] * tlwh[3] > 20 && !vertical)
        //   if (tlwh[2] * tlwh[3] > 20) {
        //     cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
        //     cv::putText(image, cv::format("%d. %s", output_stracks[i].track_id, label.c_str()),
        //                 cv::Point(tlwh[0], tlwh[1] - 5), 0, 0.6, cv::Scalar(0, 0, 255), 2,
        //                 cv::LINE_AA);
        //     cv::rectangle(image, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
        //   }
        // }
        model.draw(image, output_stracks);
        // Log detected objects
        for (const auto &obj : res) {
            auto box = obj.bbox;
            auto class_id = obj.class_id;
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

                if (obj.class_id == 2 || obj.class_id == 4 ||
                    obj.class_id == 5)  // only consider car/bus/truck
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

            if (class_id >= 12 && class_id <= 17 && conf > 0.9) {
                int newSpeed = (class_id - 9) * 10;  // class_id 12 -> 30km/h, 13 -> 40, etc.
                maxSpeed = newSpeed;
            }
        }

        laneDetector.drawLanes(image, lanes);
        // FPS calculation
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
        // Draw FPS
        cv::putText(image, cv::format("FPS: %.2f", fps), cv::Point(10, 30),
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