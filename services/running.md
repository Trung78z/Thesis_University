trtexec --onnx=best.onnx --saveEngine=best.engine
<!-- trtexec --onnx=best.onnx --saveEngine=best.engine --explicitBatch -->
yolo export model=best.pt format=onnx dynamic=False opset=11


/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=car_d.engine --fp16

yolo export model=best.pt format=onnx

check same same version tensorRT