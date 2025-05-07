

def calculate_performance_metrics(video_length, frame_size, model_name, num_params,
    avg_fps, avg_inference_time, total_inference_time,
    total_objects_detected, video_name, total_frames,
    avg_objects_per_frame, max_objects_detected):
    return {
    "Metric": [
    "Video Name",
    "Video Length (frames)",
    "Frame Size",
    "Model",
    "Parameters",
    "Average FPS",
    "Inference Time per Frame (ms)",
    "Average Objects per Frame",
    "Max Objects Detected"
    ],
    "Value": [
    video_name,
    total_frames,
    f"{frame_size[0]}x{frame_size[1]}",
    model_name,
    f"{num_params:,}",
    f"{avg_fps:.2f}",
    f"{avg_inference_time * 1000:.2f}",
    f"{avg_objects_per_frame:.2f}",
    max_objects_detected
    ]
    }