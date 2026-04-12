# src/model/guardrails.py
import numpy as np
from PIL import Image


class GuardrailError(ValueError):
    """Raised when an image fails input validation."""
    pass


MIN_DIM = 100
MAX_DIM = 4096
BLANK_THRESHOLD = 5       # mean pixel value below this → blank
OVEREXPOSED_THRESHOLD = 250  # mean pixel value above this → overexposed
MAX_DETECTIONS = 150


def validate_input(img: Image.Image) -> None:
    """
    Validate image before inference. Raises GuardrailError with a
    descriptive message if the image should not be processed.
    """
    if img.mode != "RGB":
        raise GuardrailError(f"Image must be RGB, got {img.mode}")

    w, h = img.size
    if min(w, h) < MIN_DIM:
        raise GuardrailError(f"Image too small: {w}x{h} (minimum {MIN_DIM}px on shortest edge)")
    if max(w, h) > MAX_DIM:
        raise GuardrailError(f"Image too large: {w}x{h} (maximum {MAX_DIM}px on longest edge)")

    mean_val = np.array(img).mean()
    if mean_val < BLANK_THRESHOLD:
        raise GuardrailError(f"Image appears blank (mean pixel value: {mean_val:.1f})")
    if mean_val > OVEREXPOSED_THRESHOLD:
        raise GuardrailError(f"Image appears overexposed (mean pixel value: {mean_val:.1f})")


def validate_output(
    detections: list[dict],
    confidence_threshold: float = 0.4,
) -> dict:
    """
    Filter detections by confidence and attach warnings for anomalous outputs.
    Returns dict with 'detections' (filtered) and 'warnings' (list of strings).
    """
    filtered = [d for d in detections if d["score"] >= confidence_threshold]
    warnings = []

    if len(filtered) == 0:
        warnings.append("no_detections")
    elif len(filtered) > MAX_DETECTIONS:
        warnings.append("too_many_detections")

    return {"detections": filtered, "warnings": warnings}
