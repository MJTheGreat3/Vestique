from __future__ import annotations
from PIL import Image
from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH,
    YOLO_CONF_THRESHOLD,
    CROP_PAD_FRACTION,
    ensure_yolo_weights,
)
from utils import pil_to_rgb


class ClothingDetector:
    """
    Wraps a YOLOv11 model trained on clothing/fashion data.
    Call `detect_region(image, region)` to get a clothing crop for a body region.
    """

    def __init__(self) -> None:
        ensure_yolo_weights()
        print("Loading YOLO from:", YOLO_MODEL_PATH)
        self._model = YOLO(YOLO_MODEL_PATH)

    def detect(self, image: Image.Image) -> tuple[Image.Image | None, tuple | None, float | None]:
        return self.detect_region(image, "full")

    def detect_region(
        self,
        image: Image.Image,
        region: str,  # "upper" | "lower" | "full"
    ) -> tuple[Image.Image | None, tuple | None, float | None]:
        """
        Return highest-confidence detection within the requested body region.
        Falls back to a heuristic image slice if no detection found.
        """
        img_rgb = pil_to_rgb(image)
        w, h = img_rgb.size
        region_box: tuple[int, int, int, int] = {
            "upper": (0, 0, w, int(h * 0.55)),
            "lower": (0, int(h * 0.45), w, h),
            "full":  (0, 0, w, h),
        }[region]
        results = self._model(img_rgb, conf=YOLO_CONF_THRESHOLD, verbose=False)

        best_box: tuple | None = None
        best_conf = -1.0
        rx1, ry1, rx2, ry2 = region_box

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                ix1 = max(x1, rx1)
                iy1 = max(y1, ry1)
                ix2 = min(x2, rx2)
                iy2 = min(y2, ry2)
                if ix1 >= ix2 or iy1 >= iy2:
                    continue

                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = (ix1, iy1, ix2, iy2)

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            pad_x = int((x2 - x1) * CROP_PAD_FRACTION)
            pad_y = int((y2 - y1) * CROP_PAD_FRACTION)
            padded_box = (
                max(rx1, x1 - pad_x),
                max(ry1, y1 - pad_y),
                min(rx2, x2 + pad_x),
                min(ry2, y2 + pad_y),
            )
            return img_rgb.crop(padded_box), padded_box, best_conf

        return img_rgb.crop(region_box), region_box, None
