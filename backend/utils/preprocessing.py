"""Image preprocessing helpers shared by the inference and gradcam modules."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = (224, 224)

# ImageNet mean/std (PyTorch transformers — Swin, ViT, Stage-2 Swin)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ALLOWED_EXTS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB


def validate_upload(filename: str, num_bytes: int) -> Tuple[bool, str]:
    """Cheap validation. Returns (ok, reason_if_not_ok)."""
    if not filename:
        return False, "No file provided."
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTS:
        return False, f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTS))}."
    if num_bytes > MAX_BYTES:
        return False, f"File too large ({num_bytes / 1e6:.1f} MB). Max is 10 MB."
    if num_bytes <= 0:
        return False, "Empty file."
    return True, ""


def load_image_rgb(path: str) -> np.ndarray:
    """Load image as RGB uint8 numpy array (H, W, 3). Resized to 224x224.

    Honors EXIF rotation so phone-captured fundus images aren't sideways.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.uint8)


def preprocess_tf_irv2(rgb_uint8: np.ndarray) -> np.ndarray:
    """InceptionResNetV2 preprocessing -> float32 in [-1, 1], shape (1, 224, 224, 3)."""
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

    arr = rgb_uint8.astype(np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def preprocess_tf_effnetv2(rgb_uint8: np.ndarray) -> np.ndarray:
    """EfficientNetV2 preprocessing -> float32 [-1, 1], shape (1, 224, 224, 3)."""
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

    arr = rgb_uint8.astype(np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def preprocess_torch_imagenet(rgb_uint8: np.ndarray):
    """ImageNet normalization for PyTorch transformer models.

    Returns a torch.Tensor of shape (1, 3, 224, 224), float32.
    """
    import torch

    arr = rgb_uint8.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0).float()
    return tensor
