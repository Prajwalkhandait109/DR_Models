"""GradCAM generation per model family.

* IRV2 (Keras): tf-explain GradCAM on last conv block ``conv_7b_ac``.
* Swin / Stage-2 Swin (PyTorch): pytorch-grad-cam with reshape transform.
* ViT (PyTorch): GradCAM++ with patch-grid reshape (drops CLS token).
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Optional, Tuple

import numpy as np

from model_loader import REGISTRY
from utils.preprocessing import (
    load_image_rgb,
    preprocess_tf_irv2,
    preprocess_torch_imagenet,
)
from utils.visualization import overlay_heatmap, save_png

log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "gradcam")


def _output_path() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, f"cam_{uuid.uuid4().hex}.png")


def generate_gradcam(model_key: str, image_path: str, class_idx: int) -> str:
    """Run GradCAM for ``(model_key, image_path, class_idx)`` and return the PNG path."""
    model, err = REGISTRY.get(model_key)
    if err:
        raise RuntimeError(err)

    rgb = load_image_rgb(image_path)

    if model_key == "inceptionresnetv2":
        cam = _gradcam_irv2(model, rgb, class_idx)
    elif model_key == "swin":
        cam = _gradcam_swin_torch(model, rgb, class_idx, num_classes=5)
    elif model_key == "vit":
        cam = _gradcam_vit(model, rgb, class_idx)
    elif model_key == "two_stage":
        # GradCAM only meaningful when Stage 2 ran (class_idx != 0). Fall back to Stage 2 Swin.
        stage1, stage2 = model
        if class_idx == 0:
            # Visualise Stage 1's attention instead.
            cam = _gradcam_effnet(stage1, rgb)
        else:
            cam = _gradcam_swin_torch(stage2, rgb, class_idx - 1, num_classes=4)
    else:
        raise RuntimeError(f"GradCAM not implemented for '{model_key}'.")

    overlay = overlay_heatmap(rgb, cam)
    return save_png(overlay, _output_path())


# ----------------------------- Keras (IRV2 / EffNet) ----------------------

def _gradcam_keras(model, x: np.ndarray, class_idx: Optional[int], layer_name: str) -> np.ndarray:
    """Pure tf.GradientTape GradCAM (no tf-explain dependency on layer registration)."""
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        if class_idx is None:
            loss = preds[:, 0]
        else:
            loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    cam = tf.reduce_sum(conv_out * pooled, axis=-1).numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def _gradcam_irv2(model, rgb: np.ndarray, class_idx: int) -> np.ndarray:
    x = preprocess_tf_irv2(rgb)
    return _gradcam_keras(model, x, class_idx, layer_name="conv_7b_ac")


def _gradcam_effnet(model, rgb: np.ndarray) -> np.ndarray:
    from utils.preprocessing import preprocess_tf_effnetv2

    x = preprocess_tf_effnetv2(rgb)
    # EfficientNetV2-B0 final conv block is usually 'top_conv'.
    for candidate in ("top_conv", "top_activation", "conv_head"):
        try:
            model.get_layer(candidate)
            return _gradcam_keras(model, x, None, layer_name=candidate)
        except ValueError:
            continue
    raise RuntimeError("Could not locate a final conv layer for EfficientNetV2-B0.")


# ----------------------------- Swin (torch) -------------------------------

def _swin_reshape_transform(tensor):
    # Swin output is (B, H, W, C) — pytorch-grad-cam expects (B, C, H, W).
    return tensor.permute(0, 3, 1, 2)


def _gradcam_swin_torch(model, rgb: np.ndarray, class_idx: int, num_classes: int) -> np.ndarray:
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    target_layer = model.layers[-1].blocks[-1].norm1
    device = REGISTRY.torch_device()
    x = preprocess_torch_imagenet(rgb).to(device)
    x.requires_grad_(True)

    cam_alg = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=_swin_reshape_transform,
    )
    targets = [ClassifierOutputTarget(int(class_idx))]
    grayscale_cam = cam_alg(input_tensor=x, targets=targets)[0]
    return grayscale_cam


# ----------------------------- ViT (torch) --------------------------------

def _vit_reshape_transform(tensor, height: int = 14, width: int = 14):
    # ViT tokens: (B, 197, D) -> drop CLS -> reshape to (B, D, 14, 14).
    if tensor.dim() == 3 and tensor.shape[1] == height * width + 1:
        tensor = tensor[:, 1:, :]
    b, n, d = tensor.shape
    out = tensor.reshape(b, height, width, d)
    return out.permute(0, 3, 1, 2)


def _gradcam_vit(model, rgb: np.ndarray, class_idx: int) -> np.ndarray:
    import torch
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    target_layer = model.backbone.blocks[-1].norm1
    device = REGISTRY.torch_device()
    x = preprocess_torch_imagenet(rgb).to(device)
    x.requires_grad_(True)

    cam_alg = GradCAMPlusPlus(
        model=model,
        target_layers=[target_layer],
        reshape_transform=_vit_reshape_transform,
    )
    targets = [ClassifierOutputTarget(int(class_idx))]
    grayscale_cam = cam_alg(input_tensor=x, targets=targets)[0]
    return grayscale_cam
