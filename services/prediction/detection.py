import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
import os
import time

class TensorRTDetector:
    def __init__(self, engine_path, class_names=None):
        """
        Initialize TensorRT detector with engine file
        
        Args:
            engine_path (str): Path to TensorRT engine file
            class_names (list): List of class names (e.g., COCO classes)
        """
        self.engine_path = engine_path
        self.class_names = class_names or [f"Class_{i}" for i in range(80)]  # Default to 80 classes
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        self.input_shape = self.engine.get_binding_shape(0)  # Get input shape
        self.num_classes = len(self.class_names)
        
        # Handle dynamic shapes
        if self.engine.has_dynamic_shapes:
            self.context.set_binding_shape(0, (1,) + self.input_shape[1:])
        
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        
        # Warm up the engine
        self._warm_up()
    
    def __del__(self):
        """Free CUDA memory"""
        for inp in self.inputs:
            inp['device'].free()
        for out in self.outputs:
            out['device'].free()
        self.stream.free()
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")
        
        print(f"Loading TensorRT engine from {self.engine_path}")
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def _allocate_buffers(self):
        """Allocate host and device buffers"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            binding_shape = self.engine.get_binding_shape(binding)
            if binding_shape[0] == -1:  # Dynamic batch size
                binding_shape = (1,) + binding_shape[1:]
            
            size = trt.volume(binding_shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape})
        
        return inputs, outputs, bindings, stream
    
    def _warm_up(self):
        """Warm up the engine with dummy data"""
        print("Warming up TensorRT engine...")
        dummy_input = np.random.random(self.input_shape).astype(np.float32)
        self.infer(dummy_input)
        print("Warm up complete")
    
    def preprocess(self, image):
        """
        Preprocess input image for the model
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Get input dimensions (assuming NCHW format)
        _, _, h, w = self.input_shape
        
        # Resize and normalize
        img = cv2.resize(image, (w, h))
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        
        return img
    
    def infer(self, input_data):
        """
        Run inference on input data
        
        Args:
            input_data (numpy.ndarray): Preprocessed input data
            
        Returns:
            list: Model outputs
        """
        try:
            # Ensure input data has batch dimension
            if len(input_data.shape) == 3:
                input_data = np.expand_dims(input_data, axis=0)
            
            # Set input shape for dynamic engines
            if self.engine.has_dynamic_shapes:
                self.context.set_binding_shape(0, input_data.shape)
            
            # Copy input data to host buffer
            np.copyto(self.inputs[0]['host'], input_data.ravel())
            
            # Transfer input data to the GPU
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            # Run inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # Transfer predictions back from the GPU
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            
            # Synchronize the stream
            self.stream.synchronize()
            
            return [out['host'].reshape(out['shape']) for out in self.outputs]
        except cuda.LogicError as e:
            print(f"CUDA error: {e}")
            return None
    
    def postprocess(self, outputs, original_shape, conf_thresh=0.5, iou_thresh=0.5):
        """
       =postprocess model outputs to get detections
        
        Args:
            outputs (list): Model outputs
            original_shape (tuple): Original image shape (h, w)
            conf_thresh (float): Confidence threshold
            iou_thresh (float): IoU threshold for NMS
            
        Returns:
            list: List of detections, each as [x1, y1, x2, y2, confidence, class_id]
        """
        orig_h, orig_w = original_shape
        _, _, model_h, model_w = self.input_shape
        gain = min(model_h / orig_h, model_w / orig_w)
        pad_x = (model_w - orig_w * gain) / 2
        pad_y = (model_h - orig_h * gain) / 2
        
        detections = []
        for output in outputs:
            # Assuming YOLO output: [batch, num_detections, 5 + num_classes]
            output = output.reshape(-1, 5 + self.num_classes)
            boxes = output[:, :4]  # [x_center, y_center, w, h]
            scores = output[:, 4] * np.max(output[:, 5:], axis=1)  # confidence * max_class_prob
            class_ids = np.argmax(output[:, 5:], axis=1)
            
            # Filter by confidence
            mask = scores > conf_thresh
            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]
            
            if len(boxes) > 0:
                # Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]
                boxes_xyxy = np.copy(boxes)
                boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / gain  # x1
                boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / gain  # y1
                boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2 - pad_x) / gain  # x2
                boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2 - pad_y) / gain  # y2
                
                boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, orig_w)
                boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, orig_h)
                
                # Apply NMS
                indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), conf_thresh, iou_thresh)
                indices = indices.flatten() if len(indices) > 0 else []
                
                for i in indices:
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    detections.append([x1, y1, x2, y2, scores[i], class_ids[i]])
        
        return detections
    
    def detect(self, image, conf_thresh=0.5, iou_thresh=0.5):
        """
        Perform object detection on an image
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            conf_thresh (float): Confidence threshold
            iou_thresh (float): IoU threshold for NMS
            
        Returns:
            tuple: (image with detections, list of detections)
        """
        original_shape = image.shape[:2]
        
        # Preprocess
        preprocessed = self.preprocess(image)
        
        # Add batch dimension if needed
        if len(preprocessed.shape) == 3:
            preprocessed = np.expand_dims(preprocessed, axis=0)
        
        # Run inference
        start_time = time.time()
        outputs = self.infer(preprocessed)
        if outputs is None:
            return image, []
        inference_time = time.time() - start_time
        
        # Postprocess
        detections = self.postprocess(outputs, original_shape, conf_thresh, iou_thresh)
        
        # Draw detections
        result = image.copy()
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            label = f"{self.class_names[int(class_id)]}: {conf:.2f}"
            cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(result, label, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add FPS and inference time
        fps = 1 / inference_time
        cv2.putText(result, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(f"Inference time: {inference_time * 1000:.2f} ms, FPS: {fps:.2f}")
        
        return result, detections

def main():
    # COCO class names (example)
    coco_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    # Initialize detector
    engine_path = "../runs/detect/detect/weights/best.engine"  # Replace with your engine file
    try:
        detector = TensorRTDetector(engine_path, class_names=coco_names)
    except FileNotFoundError as e:
        print(e)
        return
    
    # Load video or webcam
    video_path = "../tool/test_video.mp4"  # Replace with your video or use 0 for webcam
    if not os.path.exists(video_path) and video_path != 0:
        print(f"Video not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            result, detections = detector.detect(frame, conf_thresh=0.5, iou_thresh=0.5)
            
            # Display results
            cv2.imshow("Detection Results", result)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()