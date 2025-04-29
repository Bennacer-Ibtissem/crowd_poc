import os
import time
import numpy as np
from ultralytics import YOLO
import cv2

def get_model_info(model_path):
    """Get model size in MB and number of parameters"""
    try:
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)  # Convert to MB
        model = YOLO(model_path)
        num_params = sum(p.numel() for p in model.parameters())
        return size_mb, num_params
    except:
        return 0, 0

def process_frame(frame, model, conf_threshold, classes_to_count):
    """Process a single frame and return metrics"""
    start_time = time.time()
    
    results = model.predict(frame, conf=conf_threshold)[0]
    
    boxes = results.boxes.xyxy.cpu().numpy()
    classes_detected = results.boxes.cls.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    
    # Count people (class 0)
    object_count = sum(1 for cls in classes_detected if int(cls) in classes_to_count)
    
    # Draw bounding boxes
    for idx, cls in enumerate(classes_detected):
        if int(cls) in classes_to_count:
            x1, y1, x2, y2 = map(int, boxes[idx])
            conf = confidences[idx]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add count to frame
    cv2.putText(frame, f"Count: {object_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    return frame, object_count, inference_time

def calculate_performance_metrics(video_length, frame_size, model_name, num_params, 
                                avg_fps, avg_inference_time, total_inference_time, 
                                total_objects_detected):
    """Calculate and return performance metrics"""
    return {
        "Metric": [
            "Video Length (s)", 
            "Frame Size", 
            "Model", 
            "Parameters",
            "FPS",
            "Inference Time per Frame (ms)", 
            "Inference Time for Video (s)", 
            "True Count",
            "Predicted Count"
        ],
        "Value": [
            f"{video_length:.2f}",
            f"{frame_size[0]}x{frame_size[1]}",
            model_name,
            f"{num_params:,}",
            f"{avg_fps:.2f}",
            f"{avg_inference_time * 1000:.2f}",
            f"{total_inference_time:.2f}",
            f"{total_objects_detected}",
            f"{total_objects_detected}"
        ]
    } 
