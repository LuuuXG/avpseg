"""Locate the AVP volume from SynthSeg anatomical anchors."""

from __future__ import annotations

import numpy as np


ANCHOR_LABELS = (10, 14, 16, 28, 49, 60)


def crop_bounds(center: np.ndarray, crop_size: np.ndarray, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = np.rint(center - crop_size / 2).astype(int)
    start = np.maximum(start, 0)
    start = np.minimum(start, shape - crop_size)
    stop = start + crop_size
    return start, stop
