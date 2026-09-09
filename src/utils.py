import logging
import os
import yaml
import cv2
import numpy as np


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(name="face_recognition", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def read_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return image


def bytes_to_image(image_bytes):
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_face_box(image, bbox, label=None, color=(0, 200, 120), thickness=2):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    output = image.copy()
    cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            output, label, (x1, max(y1 - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    return output


def list_employee_folders(employee_data_dir):
    if not os.path.isdir(employee_data_dir):
        return {}
    result = {}
    for name in sorted(os.listdir(employee_data_dir)):
        folder = os.path.join(employee_data_dir, name)
        if not os.path.isdir(folder):
            continue
        images = [
            os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if images:
            result[name] = images
    return result


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
