import cv2
import numpy as np


def get_video_orientation(frame):
    """Detect video orientation based on frame dimensions"""
    if frame is None:
        return "landscape"
    height, width = frame.shape[:2]
    return "portrait" if height > width else "landscape"


def ensure_frame_dims(frame):
    """Ensure frame dimensions are valid"""
    if frame is None:
        return None
    try:
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return None
        return frame
    except:
        return None


# def enhance_frame(frame):
#     """Apply image enhancement techniques"""
#     try:
#         # Convert to LAB color space
#         lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(lab)

#         # Apply CLAHE to L channel
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#         cl = clahe.apply(l)

#         # Merge channels and convert back to BGR
#         enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

#         return enhanced
#     except Exception:
#         return frame


def handle_portrait_video(frame, force_rotate=False):
    """
    Handle portrait video orientation
    If force_rotate is True, always rotate to landscape orientation
    """
    if frame is None:
        return None

    height, width = frame.shape[:2]
    orientation = get_video_orientation(frame)

    # Only rotate if portrait and force_rotate is True
    if orientation == "portrait" and force_rotate:
        # Rotate 90 degrees clockwise to make it landscape
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return rotated

    return frame


def resize_frame(frame, target_size=(640, 640), is_portrait=False, force_rotate=False):
    """Resize frame while maintaining aspect ratio with portrait handling"""
    frame = ensure_frame_dims(frame)
    if frame is None:
        return np.zeros((*target_size, 3), dtype=np.uint8)

    # Apply portrait handling if needed
    if is_portrait:
        frame = handle_portrait_video(frame, force_rotate)

    height, width = frame.shape[:2]
    orientation = "portrait" if height > width else "landscape"

    # Store original dimensions for later use
    orig_dims = (width, height)

    # Calculate target dimensions while maintaining aspect ratio
    if orientation == "portrait":
        # For portrait videos, use height as the primary dimension
        scale = min(target_size[1] / height, target_size[0] / width)
        new_height = int(height * scale)
        new_width = int(width * scale)
    else:
        # For landscape videos, use width as the primary dimension
        scale = min(target_size[0] / width, target_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

    # Ensure minimum dimensions
    new_width = max(new_width, 32)
    new_height = max(new_height, 32)

    # Resize frame using INTER_LINEAR for better quality
    try:
        resized = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )
    except Exception:
        return np.zeros((*target_size, 3), dtype=np.uint8)

    # Create background
    background = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Calculate position to center the frame
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2

    # Ensure valid dimensions for placement
    x_offset = max(0, x_offset)
    y_offset = max(0, y_offset)

    try:
        # Place resized frame on background
        background[
            y_offset : y_offset + new_height, x_offset : x_offset + new_width
        ] = resized
    except Exception:
        return np.zeros((*target_size, 3), dtype=np.uint8)

    return background
