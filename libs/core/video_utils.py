
import cv2
import numpy as np


def get_video_orientation(frame):
    if frame is None:
        return 'landscape'
    h, w = frame.shape[:2]
    return 'portrait' if h > w else 'landscape'


def ensure_frame_dims(frame):
    try:
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return frame if h > 0 and w > 0 else None
    except:
        return None


def enhance_frame(frame):
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return frame


def resize_frame(frame, target_size=(640, 640)):
    """Resize frame while maintaining aspect ratio"""
    frame = ensure_frame_dims(frame)
    if frame is None:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    
    height, width = frame.shape[:2]
    # orientation = get_video_orientation(frame)
    
    
    # # Calculate target dimensions while maintaining aspect ratio
    # if orientation == 'portrait':
    #     # For portrait videos, use height as the primary dimension
    #     scale = min(target_size[1] / height, target_size[0] / width)
    #     new_height = int(height * scale)
    #     new_width = int(width * scale)
    # else:
    #     # For landscape videos, use width as the primary dimension
    #     scale = min(target_size[0] / width, target_size[1] / height)
    #     new_width = int(width * scale)
    #     new_height = int(height * scale)
    
    scale = min(target_size[0] / width, target_size[1] / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    # Ensure minimum dimensions
    new_width = max(new_width, 32)
    new_height = max(new_height, 32)
    
    # Resize frame using INTER_LINEAR for better quality
    try:
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    except Exception:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    
    return resized



# def resize_frame(frame, target_size=(640, 640)):
#     frame = ensure_frame_dims(frame)
#     if frame is None:
#         return np.zeros((*target_size, 3), dtype=np.uint8)

#     h, w = frame.shape[:2]

#     scale = min(target_size[0] / w, target_size[1] / h)
#     new_w, new_h = int(w * scale), int(h * scale)

#     resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

#     # padding
#     pad_w = target_size[0] - new_w
#     pad_h = target_size[1] - new_h

#     top, bottom = pad_h // 2, pad_h - pad_h // 2
#     left, right = pad_w // 2, pad_w - pad_w // 2

#     return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)