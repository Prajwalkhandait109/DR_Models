"""Visualization helpers - heatmap overlay & PNG saving."""

from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image


def overlay_heatmap(rgb_uint8: np.ndarray, cam01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a [0,1] GradCAM heatmap on an RGB image. Returns RGB uint8 (H,W,3)."""
    if cam01.ndim != 2:
        raise ValueError("cam01 must be a 2D array")
    # Resize cam to image size
    h, w = rgb_uint8.shape[:2]
    cam = cv2.resize(cam01.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    cam = np.clip(cam, 0.0, 1.0)
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap_rgb + (1.0 - alpha) * rgb_uint8.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_png(rgb_uint8: np.ndarray, path: str) -> str:
    """Save an RGB array as PNG. Returns the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb_uint8).save(path, format="PNG", optimize=True)
    return path
