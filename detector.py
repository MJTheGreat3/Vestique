from __future__ import annotations
import numpy as np
from PIL import Image
from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH,
    YOLO_POSE_MODEL,
    YOLO_CONF_THRESHOLD,
    CROP_PAD_FRACTION,
    HIP_CONF_THRESHOLD,
    ensure_yolo_weights,
)
from utils import pil_to_rgb


class ClothingDetector:
    """
    Combines YOLO person detection + YOLO-pose keypoints for
    anatomically accurate upper / lower / full body cropping.
    """

    def __init__(self) -> None:
        print("Loading YOLO detector from:", YOLO_MODEL_PATH)
        self._model = YOLO(YOLO_MODEL_PATH)
        print("Loading YOLO pose from:", YOLO_POSE_MODEL)
        self._pose = YOLO(YOLO_POSE_MODEL)

    @staticmethod
    def _get_hip_y(pose_result) -> int | None:
        """Return mean y of visible hips (keypoints 11, 12), or None."""
        kps = pose_result.keypoints
        if kps is None:
            return None
        xy = kps.xy[0].cpu().numpy()   # (17, 2)
        conf = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(17)
        hip_ys = []
        for idx in (11, 12):
            if conf[idx] >= HIP_CONF_THRESHOLD:
                hip_ys.append(int(xy[idx, 1]))
        if len(hip_ys) < 2:
            return None
        return int(np.mean(hip_ys))

    def detect_region(
        self,
        image: Image.Image,
        region: str,
    ) -> tuple[Image.Image | None, tuple | None, float | None]:
        """
        1. Find highest-confidence person (class 0) via YOLO.
        2. Get hip y via YOLO-pose.
        3. Derive region box:
            upper  – person top → hip y
            lower  – hip y → person bottom
            full   – person box as-is
        4. Pad, crop, return.
        """
        img_rgb = pil_to_rgb(image)
        w, h = img_rgb.size

        # ── Step 1: person detection ────────────────────────────────────
        dets = self._model(img_rgb, conf=YOLO_CONF_THRESHOLD, verbose=False)

        person_box: tuple[int, int, int, int] | None = None
        best_conf = -1.0

        for result in dets:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:
                    continue
                conf = float(box.conf[0])
                if conf <= best_conf:
                    continue
                best_conf = conf
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                person_box = (x1, y1, x2, y2)

        # ── No person found → full-image fallback ───────────────────────
        if person_box is None:
            fallback = (0, 0, w, h)
            return img_rgb.crop(fallback), fallback, None

        px1, py1, px2, py2 = person_box

        # ── Step 2: pose → hip y ────────────────────────────────────────
        poses = self._pose(img_rgb, conf=YOLO_CONF_THRESHOLD, verbose=False)
        hip_y = None
        for pose_result in poses:
            hip_y = self._get_hip_y(pose_result)
            if hip_y is not None:
                break

        # ── Step 3: region box ──────────────────────────────────────────
        if hip_y is None:
            # fallback proportional split within person box
            split_y = py1 + int((py2 - py1) * 0.5)
        else:
            split_y = hip_y

        region_box: tuple[int, int, int, int] = {
            "upper": (px1, py1, px2, split_y),
            "lower": (px1, split_y, px2, py2),
            "full":  (px1, py1, px2, py2),
        }[region]

        # ── Step 4: pad & crop ──────────────────────────────────────────
        rx1, ry1, rx2, ry2 = region_box
        pad_x = int((rx2 - rx1) * CROP_PAD_FRACTION)
        pad_y = int((ry2 - ry1) * CROP_PAD_FRACTION)
        padded_box = (
            max(0, rx1 - pad_x),
            max(0, ry1 - pad_y),
            min(w, rx2 + pad_x),
            min(h, ry2 + pad_y),
        )

        return img_rgb.crop(padded_box), padded_box, best_conf
