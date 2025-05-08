import cv2
import numpy as np
from time import time
from libs.core.video_utils import resize_frame, ensure_frame_dims
from libs.core.visualization import draw_boxes, DEFAULT_CLASS_MAPPING


def process_frame_object(
    frame, model, conf_threshold, classes_to_count, class_mapping=None
):
    """
    Enhanced process_frame_object with better error handling and scaling

    Args:
        frame: Input frame
        model: YOLO model
        conf_threshold: Confidence threshold
        classes_to_count: List of class IDs to count
        class_mapping: Dictionary mapping class IDs to class names (optional)
    """
    start_time = time()

    # Use default mapping if not provided
    if class_mapping is None:
        class_mapping = DEFAULT_CLASS_MAPPING

    # Ensure valid frame
    frame = ensure_frame_dims(frame)
    if frame is None:
        return np.zeros((640, 640, 3), dtype=np.uint8), {}, 0

    # Preprocess the frame
    orig_height, orig_width = frame.shape[:2]
    resized = resize_frame(frame)

    try:
        # Run YOLO prediction
        results = model.predict(
            resized,
            conf=conf_threshold,
            iou=0.45,
            classes=classes_to_count,
            verbose=False,
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
    except Exception as e:
        # Return empty results on error
        return frame, {}, time() - start_time

    # Count objects by class using np.unique for efficiency
    object_count = {}
    if len(class_ids) > 0:
        # Count occurrences of each class ID
        unique_ids, counts = np.unique(class_ids, return_counts=True)
        # fill the object_count dictionary with counts for specified classes
        for i in classes_to_count:
            if i in class_mapping:
                class_name = class_mapping[i]
                # check if the class ID exists in unique_ids
                # and get the corresponding count
                idx = np.where(unique_ids == i)[0]
                if len(idx) > 0:
                    object_count[class_name] = int(counts[idx[0]])
                else:
                    object_count[class_name] = 0

    # Draw boxes with detailed information
    display_frame = draw_boxes(
        frame.copy(),
        boxes,
        class_ids,
        confidences,
        orig_width,
        orig_height,
        resized.shape[1],
        resized.shape[0],
        classes_to_count,
        object_count,
        class_mapping,
    )

    # Return processed frame, counts dictionary, and inference time
    return display_frame, object_count, time() - start_time
