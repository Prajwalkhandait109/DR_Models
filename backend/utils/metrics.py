"""Static metadata about the four models — surfaced to the UI."""

from __future__ import annotations

# Five-class severity labels in canonical order (used by IRV2, Swin, ViT, Two-Stage final output).
CLASS_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

# Two-Stage Stage-2 classes (No DR is decided by Stage 1).
CLASS_NAMES_STAGE2 = ["Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

# Threshold below which Stage 1 short-circuits with "No DR".
TWO_STAGE_NO_DR_THRESHOLD = 0.30

# Severity badge color (hex) per class index.
SEVERITY_COLORS = {
    0: "#27AE60",  # No DR — green
    1: "#F1C40F",  # Mild — yellow
    2: "#E67E22",  # Moderate — orange
    3: "#E74C3C",  # Severe — red
    4: "#922B21",  # Proliferative — deep red
}

# Short clinical recommendation per class.
RECOMMENDATIONS = {
    0: "No signs of diabetic retinopathy detected. Annual screening recommended.",
    1: "Mild non-proliferative DR. Re-examination in 6–12 months recommended.",
    2: "Moderate non-proliferative DR. Referral to an ophthalmologist within 3–6 months.",
    3: "Severe non-proliferative DR. Urgent ophthalmology referral required.",
    4: "Proliferative DR. Immediate ophthalmology referral and treatment required.",
}

# Each model's metadata used by the UI cards and badges. Numbers sourced from
# blackbook Table 5.11 (real values from the notebooks).
MODELS = {
    "inceptionresnetv2": {
        "name": "InceptionResNetV2",
        "framework": "TensorFlow / Keras",
        "tagline": "Hybrid CNN — Inception modules + residual connections",
        "val_accuracy": 0.7975,
        "macro_f1": 0.56,
        "weighted_f1": 0.77,
        "params_m": 55.9,
        "weights_file": "inceptionresnetv2_dr_model.h5",
    },
    "swin": {
        "name": "Swin Transformer",
        "framework": "PyTorch / timm",
        "tagline": "Hierarchical shifted-window self-attention",
        "val_accuracy": 0.8347,
        "macro_f1": 0.71,
        "weighted_f1": 0.83,
        "params_m": 87.8,
        "weights_file": "swin_transformer_dr.pth",
    },
    "vit": {
        "name": "Vision Transformer",
        "framework": "PyTorch / timm",
        "tagline": "Pure transformer — global self-attention on 16×16 patches",
        "val_accuracy": 0.8540,
        "macro_f1": 0.72,
        "weighted_f1": 0.85,
        "params_m": 86.2,
        "weights_file": "vit_checkpoint.pth",
        "best": True,
    },
    "two_stage": {
        "name": "Two-Stage Screening",
        "framework": "EfficientNetV2B0 + Swin",
        "tagline": "Clinical cascade — binary screen + 4-class grader",
        "val_accuracy": 0.7000,  # Stage 2 only (4-class)
        "macro_f1": 0.61,
        "weighted_f1": 0.69,
        "params_m": 7.1,  # Stage 1 size; Stage 2 adds ~88M
        "weights_file": "efficientnetv2b0_binary_dr.h5 + complete_swin_model (1).pth",
    },
}
