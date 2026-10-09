"""Per-frame target boxes from mask files."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


def frame_index(path: Path) -> int:
    return int(re.findall(r"\d+", path.stem)[-1])


def load_boxes(mask_dir: str, pattern: str, label: int | None, limit: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Frames with a non-empty target mask and their boxes [x0, y0, x1, y1] (pixel edges).

    `label` selects one id from a label map; None (or a binary mask) takes every nonzero pixel.
    """
    frames, boxes = [], []
    for path in sorted(Path(mask_dir).glob(pattern), key=frame_index)[:limit]:
        raw = np.squeeze(np.load(path)) if path.suffix == ".npy" else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.ndim != 2:
            continue
        ys, xs = np.nonzero(raw == label if label is not None and raw.max() > 1 else raw > 0)
        if xs.size:
            frames.append(frame_index(path))
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.asarray(frames, dtype=int), np.asarray(boxes, dtype=float).reshape(-1, 4)
