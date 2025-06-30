#include <process.hpp>

/**
 * @brief Setting up Tensorrt logger
 */
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
} logger;

int main(int argc, char **argv) {
    try {
        cxxopts::Options options("test", "Run inference on a video or images (choose only one)");

        options.add_options()("v,video", "Video path", cxxopts::value<std::string>())(
            "i,images", "Images path", cxxopts::value<std::string>())(
            "m,engine", "Engine path", cxxopts::value<std::string>())("h,help", "Print usage");

        auto result = options.parse(argc, argv);

        // Print usage instructions if --help is provided
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        std::string videoPath, imagePath, enginePath;

        // Check required parameters
        if (result.count("engine")) {
            enginePath = result["engine"].as<std::string>();
        } else {
            std::cerr << "❌ Error: --engine is required.\n";
            return 1;
        }

        if (result.count("video")) videoPath = result["video"].as<std::string>();
        if (result.count("images")) imagePath = result["images"].as<std::string>();

        // Ensure only one of --video or --images is provided
        if (!videoPath.empty() && !imagePath.empty()) {
            std::cerr << "❌ Error: Please provide either --video or --images, not both.\n";
            return 1;
        }

        if (videoPath.empty() && imagePath.empty()) {
            std::cerr << "❌ Error: You must provide either --video or --images.\n";
            return 1;
        }

        // Do not run if the file is .onnx
        if (enginePath.find(".onnx") != std::string::npos) {
            std::cout << "ℹ️ ONNX model detected, skipping inference.\n";
            return 0;
        }

        // Load model
        std::cout << "🔧 Loading engine from: " << enginePath << std::endl;
        Detect model(enginePath, logger);

        // Run inference based on input type
        if (!videoPath.empty()) {
            std::cout << "🎞️ Running video inference on: " << videoPath << std::endl;
            if (checkVideo(videoPath)) {
                return runVideo(videoPath, model);
            } else {
                std::cerr << "❌ Invalid video path.\n";
                return 0;
            }
        }

        if (!imagePath.empty()) {
            std::vector<std::string> imageList;
            std::cout << "🖼️ Running image inference in folder: " << imagePath << std::endl;
            if (checkImages(imagePath, imageList)) {
                return runImages(imageList, model);
            } else {
                std::cerr << "❌ No valid images found in: " << imagePath << std::endl;
                return 0;
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
