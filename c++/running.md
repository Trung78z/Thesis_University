sudo apt-get update
sudo apt-get install libspdlog-dev

sudo apt-get install libfmt-dev

<!-- trtexec --onnx=best.onnx --saveEngine=best.engine --explicitBatch -->
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best2.engine --fp16


/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=car_detect.engine --fp16  --explicitBatch


/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=car_detect.engine --fp16 --verbose




#Check using gpu
```
sudo tegrastats
```


```
export CUDA_HOME=/usr/local/cuda
export PATH=$(CUDA_HOME)/bin:$(PATH)
export LD_LIBRARY_PATH=$(CUDA_HOME)/lib64:$(LD_LIBRARY_PATH)
```


scp -r ./models trung@192.168.1.8:~/Desktop/c++ 


sudo apt-get install libcanberra-gtk-module libcanberra-gtk3-module


CMake: Delete Cache and Reconfigure