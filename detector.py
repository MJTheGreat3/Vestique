from __future__ import annotations
import numpy as np
from PIL import Image
from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH,
    YOLO_CONF_THRESHOLD,
    CROP_PAD_FRACTION,
    ensure_yolo_weights,
)
from utils import pad_crop, pil_to_rgb


class ClothingDetector:
    """
    Wraps a YOLOv8 model trained on clothing/fashion data.
    Call `detect(image)` to get the primary clothing crop.
    """

    def __init__(self) -> None:
        ensure_yolo_weights()
        self._model = YOLO(YOLO_MODEL_PATH)

    def detect(self, image: Image.Image) -> tuple[Image.Image | None, tuple | None, float | None]:
        """
        Run inference and return the highest-confidence clothing detection.

        Returns
        -------
        crop        : PIL.Image or None  – the cropped region
        box         : (x1, y1, x2, y2) or None
        confidence  : float or None
        """
        img_rgb = pil_to_rgb(image)
        results  = self._model(img_rgb, conf=YOLO_CONF_THRESHOLD, verbose=False)

        best_box  = None
        best_conf = -1.0

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box  = tuple(int(v) for v in box.xyxy[0].tolist())

        if best_box is None:
            return None, None, None

        crop = pad_crop(img_rgb, best_box, pad=CROP_PAD_FRACTION)
        return crop, best_box, best_conf