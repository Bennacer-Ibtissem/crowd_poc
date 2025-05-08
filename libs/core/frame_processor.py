import cv2
import numpy as np
from time import time
from libs.core.video_utils import (
    resize_frame,
    get_video_orientation,
    ensure_frame_dims,
)


# Default class mapping for common COCO classes
DEFAULT_CLASS_MAPPING = {
    0: "human",  # person
    2: "car",  # car
    5: "bus",  # bus
    7: "truck",  # truck
    1: "bicycle",  # bicycle
    3: "motorcycle",  # motorcycle
}

# Default colors for different classes
DEFAULT_COLORS = {
    0: (255, 0, 0),  # Blue for person
    2: (0, 0, 255),  # Red for car
    5: (0, 255, 0),  # Green for bus
    7: (255, 255, 0),  # Cyan for truck
    1: (255, 0, 255),  # Magenta for bicycle
    3: (0, 255, 255),  # Yellow for motorcycle
    "default": (255, 255, 255),  # White for others
}


def draw_boxes(
    frame,
    boxes,
    classes_detected,
    confidences,
    orig_w,
    orig_h,
    res_w,
    res_h,
    classes_to_count,
    counts,
    class_mapping=None,
    colors=None,
):
    """
    Draw bounding boxes on the frame with class-specific colors

    Args:
        frame: Original frame
        boxes: Bounding boxes
        classes_detected: Class IDs of detected objects
        confidences: Confidence scores
        orig_w, orig_h: Original frame dimensions
        res_w, res_h: Resized frame dimensions
        classes_to_count: List of class IDs to count
        counts: Dictionary of counts by class
        class_mapping: Dictionary mapping class IDs to names (default: DEFAULT_CLASS_MAPPING)
        colors: Dictionary mapping class IDs to colors (default: DEFAULT_COLORS)
    """
    scale_x = orig_w / res_w
    scale_y = orig_h / res_h

    # Use default mappings if none provided
    if class_mapping is None:
        class_mapping = DEFAULT_CLASS_MAPPING

    if colors is None:
        colors = DEFAULT_COLORS

    for idx, cls_id in enumerate(classes_detected):
        # Skip classes we're not counting
        if int(cls_id) not in classes_to_count:
            continue

        try:
            x1, y1, x2, y2 = boxes[idx]
            x1, y1, x2, y2 = map(
                int, [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            )

            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            conf = confidences[idx]
            color = colors.get(int(cls_id), colors["default"])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            # Add confidence score with appropriate background
            label = f"{conf:.2f}"
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            y1 = max(y1, label_size[1])

            # Draw background for text
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - baseline),
                (x1 + label_size[0], y1),
                color,
                cv2.FILLED,
            )

            # Draw text in black
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )
        except Exception:
            continue

    # Add summary text (counts) to the frame
    try:
        summary_text = ""
        for cls_id, cls_name in class_mapping.items():
            if cls_id in classes_to_count and cls_name in counts:
                if summary_text:
                    summary_text += " | "
                summary_text += f"{cls_name.capitalize()}: {counts[cls_name]}"

        if summary_text:
            # Draw text with background that spans the whole text
            font_scale = 0.7
            thickness = 1
            text_size, _ = cv2.getTextSize(
                summary_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Position at top-left with padding
            pos_x, pos_y = 10, 30

            # Draw background
            cv2.rectangle(
                frame,
                (pos_x - 5, pos_y - text_size[1] - 5),
                (pos_x + text_size[0] + 5, pos_y + 5),
                (0, 0, 0),  # Black background
                cv2.FILLED,
            )

            # Draw text
            cv2.putText(
                frame,
                summary_text,
                (pos_x, pos_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                thickness,
            )
    except Exception:
        pass

    return frame


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
            resized, conf=conf_threshold, iou=0.45, classes=classes_to_count, verbose=False
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
    except Exception as e:
        # Return empty results on error
        return frame, {}, time() - start_time

    # Count objects by class
    # Count objects by class using np.unique for efficiency
    object_count = {}
    if len(class_ids) > 0:
        # Calculer les comptes en une seule opération
        unique_ids, counts = np.unique(class_ids, return_counts=True)
 
        # Remplir object_count avec les mêmes noms de variables
        for i in classes_to_count:
            if i in class_mapping:
                class_name = class_mapping[i]
                # Vérifier si cette classe a été détectée
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
