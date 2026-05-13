"""Inference dispatch — single entrypoint ``predict`` that handles all four models."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from model_loader import REGISTRY
from utils.metrics import (
    CLASS_NAMES,
    CLASS_NAMES_STAGE2,
    RECOMMENDATIONS,
    SEVERITY_COLORS,
    TWO_STAGE_NO_DR_THRESHOLD,
)
from utils.preprocessing import (
    load_image_rgb,
    preprocess_tf_effnetv2,
    preprocess_tf_irv2,
    preprocess_torch_imagenet,
)

log = logging.getLogger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _result(class_idx: int, probs: List[float], model_key: str, *, note: str | None = None) -> Dict[str, Any]:
    return {
        "model": model_key,
        "class_idx": int(class_idx),
        "class_name": CLASS_NAMES[class_idx],
        "confidence": float(probs[class_idx]),
        "all_probs": [float(p) for p in probs],
        "class_names": CLASS_NAMES,
        "severity_color": SEVERITY_COLORS[class_idx],
        "recommendation": RECOMMENDATIONS[class_idx],
        "note": note,
    }


def predict(model_key: str, image_path: str) -> Dict[str, Any]:
    """Run inference. Raises RuntimeError if model isn't available."""
    model, err = REGISTRY.get(model_key)
    if err:
        raise RuntimeError(err)

    rgb = load_image_rgb(image_path)

    if model_key == "inceptionresnetv2":
        return _predict_irv2(model, rgb)
    if model_key == "swin":
        return _predict_swin(model, rgb)
    if model_key == "vit":
        return _predict_vit(model, rgb)
    if model_key == "two_stage":
        stage1, stage2 = model
        return _predict_two_stage(stage1, stage2, rgb)
    raise RuntimeError(f"Unhandled model '{model_key}'.")


# ------------------------- per-model -------------------------------------

def _predict_irv2(model, rgb: np.ndarray) -> Dict[str, Any]:
    x = preprocess_tf_irv2(rgb)
    out = model.predict(x, verbose=0)[0]
    probs = _softmax(out) if not np.isclose(out.sum(), 1.0, atol=1e-3) else out
    return _result(int(np.argmax(probs)), probs.tolist(), "inceptionresnetv2")


def _predict_swin(model, rgb: np.ndarray) -> Dict[str, Any]:
    import torch

    x = preprocess_torch_imagenet(rgb).to(REGISTRY.torch_device())
    with torch.no_grad():
        logits = model(x)[0].detach().cpu().numpy()
    probs = _softmax(logits)
    return _result(int(np.argmax(probs)), probs.tolist(), "swin")


def _predict_vit(model, rgb: np.ndarray) -> Dict[str, Any]:
    import torch

    x = preprocess_torch_imagenet(rgb).to(REGISTRY.torch_device())
    with torch.no_grad():
        logits = model(x)[0].detach().cpu().numpy()
    probs = _softmax(logits)
    return _result(int(np.argmax(probs)), probs.tolist(), "vit")


def _predict_two_stage(stage1, stage2, rgb: np.ndarray) -> Dict[str, Any]:
    """Stage 1 binary screener -> if P(DR) < 0.30 short-circuit to No DR."""
    import torch

    # Stage 1 — EfficientNetV2B0 binary (sigmoid)
    s1_x = preprocess_tf_effnetv2(rgb)
    s1_raw = stage1.predict(s1_x, verbose=0)[0]
    p_dr = float(np.clip(s1_raw[0], 0.0, 1.0))

    if p_dr < TWO_STAGE_NO_DR_THRESHOLD:
        # 5-class output: place P(No_DR)=1-p_dr in slot 0, distribute rest uniformly.
        probs5 = [1.0 - p_dr, p_dr * 0.40, p_dr * 0.30, p_dr * 0.20, p_dr * 0.10]
        return _result(
            0,
            probs5,
            "two_stage",
            note=f"Stage 1 P(DR) = {p_dr:.3f} < {TWO_STAGE_NO_DR_THRESHOLD}. Short-circuited to No DR.",
        )

    # Stage 2 — Swin 4-class severity grader
    s2_x = preprocess_torch_imagenet(rgb).to(REGISTRY.torch_device())
    with torch.no_grad():
        s2_logits = stage2(s2_x)[0].detach().cpu().numpy()
    s2_probs = _softmax(s2_logits)  # 4-vector over Mild/Moderate/Severe/Proliferative

    # Map to 5-class output: index 0 = P(No DR) = 1 - p_dr, indices 1-4 = p_dr * s2_probs.
    probs5 = [1.0 - p_dr] + [p_dr * float(p) for p in s2_probs]
    cls_idx = int(np.argmax(probs5))
    return _result(
        cls_idx,
        probs5,
        "two_stage",
        note=f"Stage 1 P(DR) = {p_dr:.3f}. Stage 2 graded as {CLASS_NAMES_STAGE2[int(np.argmax(s2_probs))]}.",
    )
