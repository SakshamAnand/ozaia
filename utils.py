"""
OZAIANet Utility Module
Contains image validation safety gates, sample discovery, and inference preprocessing.
"""

import os
import re
from pathlib import Path
from typing import Tuple, Dict, Any, List, Union
import io
import cv2
import numpy as np
from PIL import Image


class ValidationResult:
    """Encapsulates image validation results with tuple unpacking compatibility."""

    def __init__(self, is_valid: bool, reason: str, details: Dict[str, Any] = None):
        self.is_valid = is_valid
        self.reason = reason
        self.details = details or {}

    def __iter__(self):
        # Enables unpacking: valid, reason = validate_thermal_foot_image(...)
        return iter((self.is_valid, self.reason))

    def __getitem__(self, index):
        return (self.is_valid, self.reason)[index]

    def __repr__(self):
        return f"ValidationResult(is_valid={self.is_valid}, reason='{self.reason}')"


def validate_thermal_foot_image(image_input: Union[str, Path, Image.Image, np.ndarray, io.BytesIO, Any]) -> ValidationResult:
    """
    Lightweight heuristic safety gate to verify whether an input image resembles
    the expected thermal foot-sole scan characteristics before running OZAIANet.

    Validation criteria:
    1. Readability, valid dimensions (>=50x50), and reasonable aspect ratio.
    2. Substantial dark/black background region (overall & border margins).
    3. Single dominant foreground object (avoiding complex natural scenes).
    4. Thermal false-color palette appearance (multi-channel color variance).
    5. Foot sole contour cohesiveness (solidity & coverage).

    Returns:
        ValidationResult: Evaluates to (valid: bool, reason: str) with .details dict.
    """
    details: Dict[str, Any] = {}

    # 1. Load and normalize image to RGB numpy array
    try:
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                img_rgb = image_input.copy()
            else:
                return ValidationResult(False, "Unsupported image array shape or channel count.", details)
            pil_img = Image.fromarray(img_rgb)
        elif hasattr(image_input, "read"):
            # Supports Streamlit UploadedFile and BytesIO
            image_input.seek(0)
            pil_img = Image.open(image_input).convert("RGB")
        else:
            return ValidationResult(False, f"Unrecognized image input type: {type(image_input)}", details)
    except Exception as exc:
        return ValidationResult(False, f"Could not decode image file: {str(exc)}", details)

    img_np = np.array(pil_img)
    h, w, _ = img_np.shape
    details["dimensions"] = f"{w}x{h}"

    # Dimensional checks
    if w < 50 or h < 50:
        return ValidationResult(
            False,
            f"Image dimensions ({w}x{h}) are too small (minimum 50x50 required).",
            details
        )

    aspect_ratio = max(w, h) / min(w, h)
    details["aspect_ratio"] = round(float(aspect_ratio), 2)
    if aspect_ratio > 2.5:
        return ValidationResult(
            False,
            f"Image aspect ratio ({aspect_ratio:.2f}:1) is too elongated. Expected square or near-square plantar crop.",
            details
        )

    # Convert to grayscale for structural analysis
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 2. Dark Background Check
    # Plantar thermograms feature segmented soles isolated against a dark background
    dark_intensity_threshold = 35
    dark_mask = gray < dark_intensity_threshold
    overall_dark_ratio = float(np.mean(dark_mask))
    details["dark_background_ratio"] = round(overall_dark_ratio, 3)

    if overall_dark_ratio < 0.12:
        return ValidationResult(
            False,
            f"Insufficient dark background ({overall_dark_ratio:.1%}). Expected a segmented thermal scan against a dark background.",
            details
        )

    if overall_dark_ratio > 0.88:
        return ValidationResult(
            False,
            f"Image is predominantly empty or dark space ({overall_dark_ratio:.1%}).",
            details
        )

    # Check perimeter / border margins (should be dark borders around the foot)
    bw = max(2, int(w * 0.05))
    bh = max(2, int(h * 0.05))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:bh, :] = True
    border_mask[-bh:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True
    border_dark_ratio = float(np.mean(dark_mask[border_mask]))
    details["border_dark_ratio"] = round(border_dark_ratio, 3)

    if border_dark_ratio < 0.35:
        return ValidationResult(
            False,
            f"Image border lacks dark background margins ({border_dark_ratio:.1%}). Typical thermal foot scans are centered with dark borders.",
            details
        )

    # 3. Foreground Analysis & Dominant Component Check
    fg_mask = (gray >= dark_intensity_threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return ValidationResult(False, "No distinct foreground subject detected.", details)

    details["contour_count"] = len(contours)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = float(cv2.contourArea(largest_contour))
    total_fg_area = float(np.sum(fg_mask > 0))
    total_img_area = float(h * w)

    dominance = largest_area / total_fg_area if total_fg_area > 0 else 0.0
    fg_coverage = largest_area / total_img_area
    details["largest_contour_dominance"] = round(dominance, 3)
    details["foreground_coverage"] = round(fg_coverage, 3)

    if dominance < 0.55:
        return ValidationResult(
            False,
            f"Multiple scattered objects detected (largest represents only {dominance:.1%} of foreground). Expected one dominant foot sole.",
            details
        )

    if fg_coverage < 0.15:
        return ValidationResult(
            False,
            f"Foreground subject is too small ({fg_coverage:.1%} of frame).",
            details
        )

    # 4. Thermal False-Color Appearance Check
    # Thermal false-color palettes have high color divergence across RGB channels
    fg_pixels = img_np[fg_mask > 0]
    if len(fg_pixels) > 0:
        pixel_channel_stds = np.std(fg_pixels.astype(np.float32), axis=1)
        mean_channel_std = float(np.mean(pixel_channel_stds))
        details["thermal_color_variance"] = round(mean_channel_std, 2)

        if mean_channel_std < 22.0:
            return ValidationResult(
                False,
                f"Lacks characteristic false-color thermal palette (variance: {mean_channel_std:.1f}). Expected a color-mapped thermal scan.",
                details
            )
    else:
        return ValidationResult(False, "Could not extract foreground pixels for color analysis.", details)

    # 5. Plantar Contour Shape Characteristics
    hull = cv2.convexHull(largest_contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = largest_area / hull_area if hull_area > 0 else 0.0
    details["solidity"] = round(solidity, 3)

    if solidity < 0.65:
        return ValidationResult(
            False,
            f"Foreground contour is fragmented or non-cohesive (solidity: {solidity:.2f}). Expected a cohesive foot sole profile.",
            details
        )

    return ValidationResult(True, "Image matches expected thermal foot sole scan profile.", details)


def get_sample_images(samples_dir: str = "samples") -> List[Tuple[str, str, Path]]:
    """
    Dynamically discover the six sample images in the samples directory.
    Matches names pic1 through pic6 regardless of image extension (.png, .jpg, .webp, etc.).

    Returns:
        List of tuples: (display_label, filename, Path_object) sorted pic1 to pic6.
    """
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    samples_path = Path(samples_dir)

    if not samples_path.exists() or not samples_path.is_dir():
        return []

    # Map slot index 1..6 to matched file path
    detected: Dict[int, Path] = {}
    pattern = re.compile(r"^pic([1-6])(\..+)?$", re.IGNORECASE)

    for file_path in samples_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            match = pattern.match(file_path.stem + file_path.suffix) or pattern.match(file_path.name)
            if match:
                idx = int(match.group(1))
                detected[idx] = file_path

    # Build sorted list
    results = []
    for idx in range(1, 7):
        if idx in detected:
            p = detected[idx]
            results.append((f"Sample {idx} (pic{idx})", p.name, p))

    return results


def preprocess_image(image_input: Union[str, Path, Image.Image, io.BytesIO, Any]) -> np.ndarray:
    """
    Standard OZAIANet preprocessing pipeline:
    1. Load image and convert to RGB
    2. Resize to (224, 224)
    3. Convert to numpy float32
    4. Divide by 255.0
    5. Add batch dimension (1, 224, 224, 3)
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input))
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif hasattr(image_input, "read"):
        image_input.seek(0)
        img = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    img_rgb = img.convert("RGB").resize((224, 224))
    img_array = np.array(img_rgb, dtype=np.float32) / 255.0
    batch_array = np.expand_dims(img_array, axis=0)
    return batch_array


def predict_image(image_input: Union[str, Path, Image.Image, io.BytesIO, Any], model: Any) -> Tuple[float, str]:
    """
    Run OZAIANet inference on an input image.

    Returns:
        Tuple of (probability: float, predicted_class: str)
        where:
            probability > 0.5  -> 'DM'
            probability <= 0.5 -> 'CTRL'
    """
    batch_array = preprocess_image(image_input)
    raw_prediction = model.predict(batch_array, verbose=0)
    probability = float(raw_prediction[0][0])
    predicted_class = "DM" if probability > 0.5 else "CTRL"
    return probability, predicted_class
