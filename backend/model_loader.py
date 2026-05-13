"""Lazy loader / registry for the four DR models.

A model is loaded on first request and cached in memory. If its weights file
is missing the registry returns ``None`` so the API can respond with a
graceful "weights not yet available" message instead of crashing.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

from utils.metrics import MODELS

log = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def _weights_path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


TWO_STAGE_STAGE1_FILE = "efficientnetv2b0_binary_dr.h5"
TWO_STAGE_STAGE2_FILE = "complete_swin_model (1).pth"


def has_weights(model_key: str) -> bool:
    if model_key == "two_stage":
        return os.path.exists(_weights_path(TWO_STAGE_STAGE1_FILE)) and os.path.exists(
            _weights_path(TWO_STAGE_STAGE2_FILE)
        )
    fname = MODELS[model_key]["weights_file"]
    return os.path.exists(_weights_path(fname))


class _Registry:
    """Thread-safe lazy model registry."""

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._device = None  # torch device, set on first PyTorch load

    # --------------------------- public API -----------------------------

    def get(self, key: str) -> Tuple[Optional[Any], Optional[str]]:
        """Returns (model, error). Either model is non-None or error explains why not."""
        if key not in MODELS:
            return None, f"Unknown model '{key}'."
        if not has_weights(key):
            return None, f"Weights for '{MODELS[key]['name']}' not yet uploaded."
        with self._lock:
            if key not in self._cache:
                try:
                    if key == "inceptionresnetv2":
                        self._cache[key] = self._load_irv2()
                    elif key == "swin":
                        self._cache[key] = self._load_swin()
                    elif key == "vit":
                        self._cache[key] = self._load_vit()
                    elif key == "two_stage":
                        self._cache[key] = self._load_two_stage()
                except Exception as exc:  # noqa: BLE001
                    log.exception("Failed to load model %s", key)
                    return None, f"Failed to load model: {exc}"
            return self._cache[key], None

    def torch_device(self):
        import torch

        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    # --------------------------- loaders --------------------------------

    def _load_irv2(self):
        from tensorflow.keras.models import load_model
        from keras.src.applications.inception_resnet_v2 import CustomScaleLayer

        path = _weights_path("inceptionresnetv2_dr_model.h5")
        log.info("Loading InceptionResNetV2 from %s", path)
        return load_model(path, compile=False, custom_objects={"CustomScaleLayer": CustomScaleLayer})

    def _load_swin(self):
        import timm
        import torch

        device = self.torch_device()
        model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=5)
        state = torch.load(_weights_path("swin_transformer_dr.pth"), map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        return model

    def _load_vit(self):
        import timm
        import torch
        import torch.nn as nn

        class ViTDRClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = timm.create_model(
                    "vit_base_patch16_224",
                    pretrained=False,
                    num_classes=0,
                    global_pool="token",
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(768),
                    nn.Dropout(0.30),
                    nn.Linear(768, 512),
                    nn.GELU(),
                    nn.Dropout(0.15),
                    nn.Linear(512, 5),
                )

            def forward(self, x):
                return self.head(self.backbone(x))

        device = self.torch_device()
        model = ViTDRClassifier()
        state = torch.load(_weights_path("vit_checkpoint.pth"), map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        return model

    def _load_two_stage(self):
        import timm
        import torch
        from tensorflow.keras.models import load_model

        # Stage 1 — Keras binary screener
        stage1 = load_model(_weights_path(TWO_STAGE_STAGE1_FILE), compile=False)

        # Stage 2 — complete Swin model saved via torch.save(model) or state_dict
        device = self.torch_device()
        stage2_path = _weights_path(TWO_STAGE_STAGE2_FILE)
        checkpoint = torch.load(stage2_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, torch.nn.Module):
            # Saved as full model object
            stage2 = checkpoint.to(device).eval()
        else:
            # Saved as state dict (with optional wrapper key)
            stage2 = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=4)
            state = checkpoint.get("model_state_dict", checkpoint)
            stage2.load_state_dict(state, strict=False)
            stage2.to(device).eval()
        return stage1, stage2


REGISTRY = _Registry()


def available_models() -> Dict[str, Dict[str, Any]]:
    """Returns the MODELS metadata dict augmented with an ``available`` flag."""
    out = {}
    for key, meta in MODELS.items():
        out[key] = dict(meta)
        out[key]["available"] = has_weights(key)
    return out
