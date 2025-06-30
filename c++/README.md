# C++ Real-time Computer Vision System

This directory contains the high-performance C++ implementation of a comprehensive computer vision system for autonomous driving applications. The system combines object detection, lane detection, object tracking, and distance estimation using YOLOv11, TensorRT, ByteTrack, and OpenCV.

## 🚀 Features

- **Real-time Object Detection**: YOLOv11 with TensorRT optimization
- **Lane Detection**: OpenCV-based lane line detection and tracking
- **Object Tracking**: ByteTrack integration for persistent object tracking
- **Distance Estimation**: Front distance calculation using camera calibration
- **Multi-modal Processing**: Combined detection and tracking pipeline
- **CUDA Acceleration**: GPU-optimized preprocessing and inference
- **Cross-platform**: Linux support with CMake build system

## 📁 Project Structure

```
c++/
├── src/                    # Source files
│   ├── main.cpp           # Application entry point
│   ├── Detect.cpp         # YOLOv11 detection implementation
│   ├── process.cpp        # Video/image processing pipeline
│   ├── utils.cpp          # Utility functions
│   └── preprocess.cu      # CUDA preprocessing kernels
├── include/               # Header files
│   ├── Detect.h           # Detection class interface
│   ├── process.hpp        # Processing pipeline interface
│   ├── utils.hpp          # Utility functions interface
│   ├── preprocess.h       # CUDA preprocessing interface
│   ├── common.h       # Global constants and configurations
│   ├── cuda_utils.h   # CUDA utility functions
│   ├── macros.h       # Common macros
│   ├── LaneDetector.h # Lane detection interface
│   ├── LaneDetector.cpp
│   ├── FrontDistanceEstimator.h
│   ├── logging.h
├── bytetrack/             # ByteTrack object tracking
│   ├── include/           # ByteTrack headers
│   └── src/               # ByteTrack implementation
├── cmake/                 # CMake configuration files
│   ├── FindCUDAConfig.cmake
│   ├── FindOpenCVConfig.cmake
│   └── FindTensorRTConfig.cmake
├── models/                # Pre-trained models
├── bin/                   # Compiled executables
├── build/                 # Build directory
├── images/                # Test images
├── videos/                # Test videos
├── CMakeLists.txt         # Main CMake configuration
├── Makefile               # Build automation
└── running.md             # Quick start guide
```

## 🛠️ Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (Compute Capability 7.0+)
- **Memory**: Minimum 4GB GPU memory (8GB recommended)
- **RAM**: 8GB system RAM minimum

### Software Requirements
- **OS**: Ubuntu 20.04 or later
- **CUDA**: 11.8 or later
- **cuDNN**: 8.6 or later
- **TensorRT**: 8.6 or later
- **OpenCV**: 4.5 or later
- **CMake**: 3.10 or later
- **GCC**: 7.5 or later

### System Dependencies
```bash
sudo apt-get update
sudo apt-get install libspdlog-dev libfmt-dev
sudo apt-get install libcanberra-gtk-module libcanberra-gtk3-module
sudo apt-get install libopencv-dev libopencv-contrib-dev
```

## 🔧 Installation & Build

### 1. Environment Setup
```bash
# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 2. Build Options

#### Option A: Using Makefile (Recommended)
```bash
# Build and run with default settings
make run

# Build only
make build

# Clean build
make clean

# Debug build
make CMAKE_BUILD_TYPE=Debug build
```

#### Option B: Using CMake directly
```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

### 3. Build Configuration

The build system supports several configuration options:

```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..

# Custom CUDA architecture
cmake -DCUDA_ARCH_LIST="7.5;8.0;8.6" ..
```

## 🎯 Usage

### Basic Usage

#### Object Detection on Video
```bash
./bin/detection --engine models/best.engine --video path/to/video.mp4
```

#### Object Detection on Images
```bash
./bin/detection --engine models/best.engine --images path/to/images/
```

#### Webcam Detection
```bash
./bin/detection --engine models/best.engine --video 0
```

### Command Line Options

```bash
./bin/detection --help
```

Available options:
- `--engine, -m`: Path to TensorRT engine file (required)
- `--video, -v`: Path to video file or camera index
- `--images, -i`: Path to image directory
- `--help, -h`: Show help message

### Model Conversion

#### Convert ONNX to TensorRT Engine
```bash
# Basic conversion
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine

# With FP16 optimization
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16

# With explicit batch
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16 --explicitBatch
```

#### Export YOLO Model to ONNX
```bash
# Using Ultralytics
yolo export model=best.pt format=onnx dynamic=False opset=11
```

## 🔍 System Components

### 1. Object Detection (`Detect` class)
- **File**: `src/Detect.cpp`, `include/Detect.h`
- **Features**:
  - TensorRT engine loading and inference
  - CUDA-accelerated preprocessing
  - Post-processing with NMS
  - Multi-class detection support
  - Configurable confidence and NMS thresholds

### 2. Lane Detection (`LaneDetector` class)
- **File**: `include/lanevision/LaneDetector.h`, `include/lanevision/LaneDetector.cpp`
- **Features**:
  - Canny edge detection
  - Hough line transform
  - Region of interest filtering
  - Lane line fitting and smoothing
  - Confidence-based filtering

### 3. Object Tracking (`ByteTrack`)
- **Directory**: `bytetrack/`
- **Features**:
  - Multi-object tracking
  - Occlusion handling
  - Motion prediction
  - Track management
  - Kalman filtering

### 4. Distance Estimation (`FrontDistanceEstimator`)
- **File**: `include/followdist/FrontDistanceEstimator.h`
- **Features**:
  - Camera calibration support
  - Object size priors
  - Focal length calculations

### 5. CUDA Preprocessing (`preprocess.cu`)
- **File**: `src/preprocess.cu`
- **Features**:
  - GPU-accelerated image resizing
  - Normalization
  - Color space conversion
  - Memory management

## 📊 Performance

### Supported Classes
The system detects 21 custom classes optimized for autonomous driving:
- **Vehicles**: car, truck, bus, motorcycle, bicycle, other-vehicle
- **Traffic Objects**: stop sign, red light, yellow light, green light
- **Pedestrians**: person
- **Infrastructure**: crosswalk
- **Speed Limits**: 30-80 km/h signs and end-of-limit signs

### Performance Metrics (GTX 1650)
| Model | Resolution | Batch | FP16 | FPS | Latency (ms) | Memory (MB) |
|-------|------------|-------|------|-----|--------------|-------------|
| YOLOv11n | 640x640 | 8 | Yes | 45 | 22.2 | 1800 |
| YOLOv11n | 640x640 | 8 | No | 35 | 28.6 | 2800 |
| YOLOv11s | 640x640 | 4 | Yes | 25 | 40.0 | 2200 |

## 🧪 Testing

### Test Images
```bash
# Run detection on test images
./bin/detection --engine models/best.engine --images images/
```

### Performance Testing
```bash
# Benchmark with specific parameters
./bin/detection --engine models/best.engine \
                --images test_images/ \
                --conf 0.25 \
                --iou 0.45
```

### GPU Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check GPU temperature
nvidia-smi -q -d temperature
```

## 🔧 Configuration

### Model Parameters
Edit `include/common/common.h` to modify:
- Input resolution (`WIDTH`, `HEIGHT`)
- Frame rate (`FPS`)
- Focal length for distance estimation
- Class names and colors
- Detection thresholds

### Detection Parameters
In `include/Detect.h`:
- `conf_threshold`: Confidence threshold (default: 0.6)
- `nms_threshold`: NMS threshold (default: 0.6)
- `num_classes`: Number of classes (default: 21)

### Lane Detection Parameters
In `include/lanevision/LaneDetector.h`:
- Canny edge detection thresholds
- Hough transform parameters
- ROI configuration
- Line filtering parameters

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Memory Errors
```bash
# Reduce batch size in CMakeLists.txt
# Or reduce input resolution in common.h
```

#### 2. TensorRT Version Mismatch
```bash
# Check TensorRT version compatibility
dpkg -l | grep tensorrt
```

#### 3. Model Loading Errors
```bash
# Verify engine file exists and is valid
ls -la models/
file models/best.engine
```

#### 4. OpenCV Errors
```bash
# Reinstall OpenCV with CUDA support
sudo apt-get install libopencv-dev libopencv-contrib-dev
```

### Performance Optimization

#### GPU Optimization
```bash
# Set maximum performance mode
sudo nvidia-smi -pm 1

# Set CPU governor to performance
sudo cpufreq-set -g performance
```

#### System Optimization
```bash
# Disable desktop effects
gsettings set org.gnome.desktop.animations enabled false

# Monitor system resources
htop
```

## 🔄 Development

### Adding New Features

#### 1. New Detection Types
- Extend the `Detect` class in `Detect.h` and `Detect.cpp`
- Add new preprocessing kernels in `preprocess.cu`
- Update class names in `common.h`

#### 2. New Tracking Features
- Modify ByteTrack implementation in `bytetrack/`
- Update tracking classes in `process.hpp`

#### 3. New Visualization
- Add drawing functions in `process.cpp`
- Update color schemes in `common.h`

### Code Style
The project uses `.clang-format` for consistent code formatting:
```bash
# Format code
clang-format -i src/*.cpp include/*.h
```

## 📚 Dependencies

### External Libraries
- **TensorRT**: NVIDIA's deep learning inference library
- **OpenCV**: Computer vision library
- **CUDA**: NVIDIA's parallel computing platform
- **Eigen3**: Linear algebra library
- **cxxopts**: Command line argument parsing

### Internal Components
- **ByteTrack**: Object tracking algorithm
- **LaneVision**: Custom lane detection module
- **FollowDist**: Distance estimation module

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Make your changes following the code style
4. Test your changes thoroughly
5. Commit your changes (`git commit -m 'Add NewFeature'`)
6. Push to the branch (`git push origin feature/NewFeature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the main [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [OpenCV](https://opencv.org/)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the main project README
3. Open an issue on GitHub with detailed information
4. Include system specifications and error messages
