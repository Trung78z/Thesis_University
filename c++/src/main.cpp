#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "Detect.h"
#include <lanevision/LaneDetector.h>

const int WIDTH = 1280;
const int HEIGHT = 720;
const int FPS = 30;

bool IsPathExist(const string &path)
{
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}

bool IsFile(const string &path)
{
    if (!IsPathExist(path))
    {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

/**
 * @brief Setting up Tensorrt logger
 */
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <engine_file_path> <image_or_video_path>" << std::endl;
        return 1;
    }

    std::cout << "Engine file path: " << argv[1] << std::endl;
    std::cout << "Input path: " << argv[2] << std::endl;

    const string engine_file_path{argv[1]};
    const string path{argv[2]};
    vector<string> imagePathList;
    bool isVideo{false};

    assert(argc == 3);
    if (IsFile(path))
    {
        string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png")
        {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv" || suffix == "webm")
        {
            isVideo = true;
        }
        else
        {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            abort();
        }
    }
    else if (IsPathExist(path))
    {
        glob(path + "/*.jpg", imagePathList);
    }

    Detect model(engine_file_path, logger);

    if (engine_file_path.find(".onnx") != std::string::npos)
    {
        return 0;
    }
    LaneDetector laneDetector;
    if (isVideo)
    {

        cout << "Opening video: " << path << endl;
        // cv::VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink', cv::CAP_GSTREAMER);
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

        if (!cap.isOpened())
        {
            cerr << "Error: Cannot open video file!" << endl;
            return 0;
        }

        // Get frame width, height, and FPS
        double fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

        int frameCount = 0;
        auto fpsStartTime = std::chrono::steady_clock::now();

        while (true)
        {
            cv::Mat image;
            cap >> image;

            if (image.empty())
            {
                break;
            }
            // Resize the image to fit the window
            cv::resize(image, image, cv::Size(WIDTH, HEIGHT));
            vector<Detection> objects;

            model.preprocess(image);

            model.infer();

            model.postprocess(objects);
            model.draw(image, objects);
            std::vector<cv::Vec4i> lanes = laneDetector.detectLanes(image);
            // if (lanes.size() >= 2)
            // {
            //     // Lấy 2 lane (trái - phải)
            //     cv::Vec4i l0 = lanes[0];
            //     cv::Vec4i l1 = lanes[1];

            //     // Tính điểm giữa lane để vẽ line
            //     cv::Point laneTop((l0[0] + l1[0]) / 2, (l0[1] + l1[1]) / 2);
            //     cv::Point laneBottom((l0[2] + l1[2]) / 2, (l0[3] + l1[3]) / 2);

            //     // Tạo polygon vùng làn theo thứ tự: top-left → top-right → bottom-right → bottom-left
            //     std::vector<cv::Point> lane_area = {
            //         cv::Point(l0[0], l0[1]), // top-left
            //         cv::Point(l1[0], l1[1]), // top-right
            //         cv::Point(l1[2], l1[3]), // bottom-right
            //         cv::Point(l0[2], l0[3])  // bottom-left
            //     };



            //     for (const auto &obj : objects)
            //     {
            //         if (obj.class_id == 2 || obj.class_id == 4 || obj.class_id == 5) // chỉ xét car/bus/truck
            //         {
            //             cv::Point foot(obj.bbox.x + obj.bbox.width / 2,
            //                            obj.bbox.y + obj.bbox.height); // đáy bbox (dùng để test)

            //             if (cv::pointPolygonTest(lane_area, foot, false) >= 0) // xe trong làn
            //             {
            //                 // ① vẽ chấm ở đáy (giữ nếu muốn)
            //                 // cv::circle(image, foot, 4, cv::Scalar(0,0,255), -1);

            //                 // ② vẽ chấm ở chính giữa bbox
            //                 cv::Point mid(obj.bbox.x + obj.bbox.width / 2,
            //                               obj.bbox.y + obj.bbox.height / 2);      // tâm bbox
            //                 cv::circle(image, mid, 5, cv::Scalar(0, 255, 0), -1); // chấm xanh lục

            //                 // (tuỳ chọn) lưu khung hình
            //                 std::string filename = "tmp/frame_in_lane_" +
            //                                        std::to_string(int(obj.conf * 1000)) + ".jpg";
            //                 cv::imwrite(filename, image);
            //             }
            //         }
            //     }
            // }
            // else
            // {
            //     std::cout << "Not enough lanes detected to define a lane area." << std::endl;
            // }

            laneDetector.drawLanes(image, lanes);
            // FPS calculation
            frameCount++;
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - fpsStartTime).count();
            if (elapsed >= 1)
            {
                fps = frameCount / static_cast<double>(elapsed);
                frameCount = 0;
                fpsStartTime = now;
            }

            // Draw FPS
            std::string fpsText = cv::format("FPS: %.2f", fps);
            cv::putText(image, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(0, 255, 0), 2);

            cv::imshow("Result", image);
            if (cv::waitKey(1) == 'q')
            { // Press ESC to exit
                break;
            }
        }

        // Release resources
        cap.release();
        cv::destroyAllWindows();
    }

    else
    {
        // path to folder saves images
        for (const auto &imagePath : imagePathList)
        {
            // open image
            Mat image = imread(imagePath);
            if (image.empty())
            {
                cerr << "Error reading image: " << imagePath << endl;
                continue;
            }

            vector<Detection> objects;
            model.preprocess(image);

            auto start = std::chrono::system_clock::now();
            model.infer();
            auto end = std::chrono::system_clock::now();

            model.postprocess(objects);
            model.draw(image, objects);
            std::vector<cv::Vec4i> lanes = laneDetector.detectLanes(image);
            laneDetector.drawLanes(image, lanes);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);

            model.draw(image, objects);

            imshow("Result", image);

            waitKey(0);
        }
    }
    return 0;
}
