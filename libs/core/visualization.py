import cv2
import numpy as np

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
