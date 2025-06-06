#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

using namespace nvinfer1;

// Function to read TensorRT engine file (*.engine)
std::vector<char> readEngineFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening engine file: " << filename << std::endl;
        return {};
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

int main() {
    // Simple logger for TensorRT
    class Logger : public ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING)
                std::cout << msg << std::endl;
        }
    } logger;

    // Read pre-built engine file (you need to have the .engine file)
    std::string engineFile = "car.engine";
    auto engineData = readEngineFile(engineFile);
    if (engineData.empty()) {
        return -1;
    }

    // Create runtime
    std::unique_ptr<IRuntime> runtime{createInferRuntime(logger)};
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return -1;
    }

    // Deserialize engine
    std::unique_ptr<ICudaEngine> engine{
        runtime->deserializeCudaEngine(engineData.data(), engineData.size())};
    if (!engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return -1;
    }

    // Create execution context
    std::unique_ptr<IExecutionContext> context{engine->createExecutionContext()};
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return -1;
    }

    // Assume model input is 1D float with size = 3, output also size = 3
    // Prepare input data
    float inputData[3] = {1.0f, 2.0f, 3.0f};
    float outputData[3] = {0};

    // Allocate GPU memory for input and output
    void* buffers[2];
    cudaMalloc(&buffers[0], sizeof(inputData));
    cudaMalloc(&buffers[1], sizeof(outputData));

    // Copy input to GPU
    cudaMemcpy(buffers[0], inputData, sizeof(inputData), cudaMemcpyHostToDevice);

    // Run inference - using enqueueV3 for newer TensorRT versions
    context->enqueueV3(0);

    // Copy result back to CPU
    cudaMemcpy(outputData, buffers[1], sizeof(outputData), cudaMemcpyDeviceToHost);

    std::cout << "Output: ";
    for (float v : outputData) std::cout << v << " ";
    std::cout << std::endl;

    // Clean up GPU memory
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    // No need to manually destroy objects - smart pointers handle it
    return 0;
}