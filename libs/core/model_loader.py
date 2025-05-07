from ultralytics import YOLO
import os


def load_yolo_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model does not exist at {model_path}")
    return YOLO(model_path)


def get_model_info(model_path: str):
    try:
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        model = YOLO(model_path)
        num_params = sum(p.numel() for p in model.parameters())
        return size_mb, num_params
    except Exception:
        return 0, 0