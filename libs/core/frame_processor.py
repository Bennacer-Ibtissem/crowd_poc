import cv2
import numpy as np
from time import time
from libs.core.video_utils import resize_frame, enhance_frame, get_video_orientation



def draw_boxes(frame, boxes, classes_detected, confidences, orig_w, orig_h, res_w, res_h, classes_to_count, counts):
    scale_x = orig_w / res_w
    scale_y = orig_h / res_h

    COLORS = {
        0: (255,0,0),
        2: (0, 0, 255),
        5: (0, 255, 0),
        'default': (255, 255, 0)
    }

    for idx, cls_id in enumerate(classes_detected):
        if int(cls_id) not in classes_to_count:
            continue
        if int(cls_id) in classes_to_count:
            x1, y1, x2, y2 = boxes[idx]
            x1, y1, x2, y2 = map(int, [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
            conf = confidences[idx]
            color = COLORS.get(int(cls_id), COLORS['default'])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{conf:.2f}"
            
            # # class_name = "Car" if int(cls_id) == 2 else "Bus" if int(cls_id) == 5 else str(int(cls_id))
            # label = f"{class_name} - {vehicle_color.capitalize()}"
            # label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            # y_label = max(y1, label_size[1] + baseline)
            # cv2.rectangle(frame, (x1, y_label - label_size[1] - baseline),
            #             (x1 + label_size[0], y_label), color, cv2.FILLED)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    
    # summary_text = f"Pedestrians: {counts.get('human', 0)} | Cars: {counts.get('car', 0)} | Buses: {counts.get('bus', 0)}"
    summary_text = ''
    for key, value in counts.items():
        if key == 'human':
            summary_text = f" Pedestrians: {value}"
        elif key == 'car':
            summary_text += f" Cars: {value}"
        elif key == 'bus':
            summary_text += f" Buses: {value}"
    
    cv2.putText(frame, summary_text, (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 5)

    return frame



def process_frame_object(frame, model, conf_threshold, classes_to_count):
    start_time = time()
    if frame is None:
        return frame, 0, 0

    orig_height, orig_width = frame.shape[:2]
    enhanced = enhance_frame(frame)
    resized = resize_frame(enhanced)

    try:
        results = model.predict(resized, conf=conf_threshold, iou=0.45, classes=classes_to_count)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
    except Exception:
        return frame, 0, time() - start_time

    # object_count = len(class_ids)
    object_count = {}
    dic = {0: 'human', 2: 'car', 5: 'bus'}
    for i in classes_to_count:
        object_count[dic[i]] = int(np.sum(class_ids == i))
 

    display_frame = draw_boxes(frame.copy(), boxes, class_ids, confidences, orig_width, orig_height, resized.shape[1], resized.shape[0], classes_to_count, object_count)
    return display_frame, object_count, time() - start_time