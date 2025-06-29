# Real-time Object Detection and Lane Detection using YOLOv11 and TensorRT

This repository contains the implementation of a comprehensive real-time computer vision system for autonomous driving applications. The project combines object detection, lane detection, and object tracking using YOLOv11, TensorRT, and ByteTrack.

## Overview

The project implements a multi-modal computer vision pipeline that leverages:
- **YOLOv11** for state-of-the-art object detection
- **NVIDIA TensorRT** for optimized inference
- **ByteTrack** for robust object tracking
- **OpenCV-based lane detection** for road lane identification
- **CUDA acceleration** for GPU-based processing
- **ROS integration** for robotic applications

## Features

- **Multi-modal Detection**: Object detection, lane detection, and distance estimation
- **Real-time Performance**: Optimized for real-time video processing
- **Object Tracking**: ByteTrack integration for persistent object tracking
- **Multiple Input Sources**: Support for video files, webcam, and RTSP streams
- **Cross-platform**: C++ implementation with Python services
- **ROS Integration**: Ready for robotic applications
- **Performance Monitoring**: Built-in performance benchmarking tools

## Project Structure

```
├── c++/                    # Main C++ implementation
│   ├── src/               # Source code
│   │   ├── main.cpp       # Main application entry point
│   │   ├── Detect.cpp     # YOLOv11 detection implementation
│   │   ├── process.cpp    # Video/image processing pipeline
│   │   └── preprocess.cu  # CUDA preprocessing kernels
│   ├── include/           # Header files
│   │   ├── Detect.h       # Detection class interface
│   │   ├── lanevision/    # Lane detection module
│   │   ├── followdist/    # Distance estimation
│   │   └── tensorrt/      # TensorRT utilities
│   ├── models/            # Pre-trained models
│   ├── bin/               # Compiled executables
│   └── CMakeLists.txt     # Build configuration
├── services/              # Python services
│   ├── models/            # Python model files
│   ├── prediction/        # Prediction services
│   └── requirements.txt   # Python dependencies
├── ros/                   # ROS package
│   ├── src/              # ROS nodes
│   └── package.xml       # ROS package configuration
├── notebooks/             # Training notebooks
│   ├── train.ipynb       # YOLOv11 training workflow
│   └── thesis-train-all.ipynb
└── vision_opencv/         # OpenCV vision utilities
```

## Requirements

### Hardware Requirements
- NVIDIA GPU with CUDA support (Compute Capability 7.0 or higher)
- Minimum 4GB GPU memory (8GB recommended)
- Sufficient system RAM (8GB or more)

### Software Requirements
- Ubuntu 20.04 or later
- CUDA 11.8 or later
- cuDNN 8.6 or later
- TensorRT 8.6 or later
- Python 3.8 or later
- OpenCV 4.5 or later
- CMake 3.10 or later

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Trung78z/Thesis_University.git
cd Thesis_University
git submodule update --init --recursive
```

### 2. Install System Dependencies
```bash
sudo apt-get update
sudo apt-get install libspdlog-dev libfmt-dev
sudo apt-get install libcanberra-gtk-module libcanberra-gtk3-module
```

### 3. Set up CUDA Environment
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 4. Build C++ Implementation
```bash
cd c++
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 5. Install Python Dependencies
```bash
cd ../services
python3 -m venv .env --system-site-packages
source .env/bin/activate
pip install -r requirements.txt
```

## Usage

### C++ Implementation

#### Basic Object Detection
```bash
# Run detection on video
./bin/detection --engine models/best.engine --video path/to/video.mp4

# Run detection on images
./bin/detection --engine models/best.engine --images path/to/images/
```

#### Model Conversion
```bash
# Convert ONNX to TensorRT engine
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16

# Export YOLO model to ONNX
yolo export model=best.pt format=onnx dynamic=False opset=11
```

### Python Services

#### Training YOLOv11 Model
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo11n.pt")

# Train on custom dataset
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="auto",
    lr0=0.005
)
```

#### Inference Service
```python
# Load trained model
model = YOLO('best.pt')

# Run inference
results = model('path/to/image.jpg')
```

### ROS Integration

```bash
# Build ROS package
cd ros
catkin_make

# Launch the system
roslaunch thesis thesis.launch
```

## Model Information

### Supported Classes
The system detects 80 COCO classes including:
- **Vehicles**: car, truck, bus, motorcycle, bicycle
- **Traffic Objects**: traffic light, stop sign, parking meter
- **Pedestrians**: person
- **Animals**: Various animals (cat, dog, horse, etc.)
- **Objects**: Various everyday objects

### Model Variants
- **YOLOv11n**: Nano model (~2.6M parameters, ~6.5 GFLOPs)
- **YOLOv11s**: Small model
- **YOLOv11m**: Medium model
- **YOLOv11l**: Large model

## Performance

### Inference Speed (NVIDIA GTX 1650)
| Model    | Resolution | Batch | FP16 | FPS  | Latency (ms) |
|----------|------------|-------|------|------|--------------|
| YOLOv11n | 640x640    | 8     | Yes  | 45   | 22.2        |
| YOLOv11n | 640x640    | 8     | No   | 35   | 28.6        |
| YOLOv11s | 640x640    | 4     | Yes  | 25   | 40.0        |

### Memory Usage
| Model    | Batch | FP16 | VRAM (MB) | CPU RAM (MB) |
|----------|-------|------|-----------|--------------|
| YOLOv11n | 8     | Yes  | 1800      | 1200         |
| YOLOv11s | 4     | Yes  | 2200      | 1500         |

## Advanced Features

### Lane Detection
The system includes a robust lane detection module using:
- Canny edge detection
- Hough line transform
- Region of interest (ROI) filtering
- Lane line fitting and smoothing
- Confidence-based filtering

### Object Tracking
ByteTrack integration provides:
- Persistent object tracking across frames
- Occlusion handling
- Motion prediction
- Track management

### Distance Estimation
Front distance estimation using:
- Camera calibration
- Object size priors
- Focal length calculations

## Development

### Training Custom Models
1. Prepare dataset in YOLO format
2. Create `data.yaml` configuration
3. Use the provided Jupyter notebooks for training
4. Export to ONNX format
5. Convert to TensorRT engine

### Adding New Features
- Extend the `Detect` class for new detection types
- Add new preprocessing kernels in `preprocess.cu`
- Implement new ROS nodes for additional functionality

## Testing

### Test Images
The system includes test images for evaluation:
```
test_images/
├── day/           # Daytime scenarios
├── night/         # Night scenarios
└── challenging/   # Challenging conditions
```

### Performance Testing
```bash
# Run benchmark
./bin/detection --engine models/best.engine \
                --images test_images/ \
                --conf 0.25 \
                --iou 0.45
```

## Troubleshooting

### Common Issues
1. **CUDA Memory Errors**: Reduce batch size or input resolution
2. **TensorRT Version Mismatch**: Ensure TensorRT version compatibility
3. **Model Loading Errors**: Check engine file path and format
4. **Performance Issues**: Enable FP16, adjust batch size, check GPU utilization

### Performance Optimization
1. **GPU Optimization**:
   ```bash
   sudo nvidia-smi -pm 1  # Set maximum performance mode
   sudo cpufreq-set -g performance  # Set CPU governor
   ```

2. **System Optimization**:
   ```bash
   gsettings set org.gnome.desktop.animations enabled false
   ```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [OpenCV](https://opencv.org/)

## Contact

For questions and support, please open an issue on GitHub.

Project Link: [https://github.com/Trung78z/Thesis_University]
