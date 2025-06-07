sudo apt-get update
sudo apt-get install libspdlog-dev

sudo apt-get install libfmt-dev

<!-- trtexec --onnx=best.onnx --saveEngine=best.engine --explicitBatch -->
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=car_detect.engine --fp16