"""Shared data and calls for expertise tests."""

import numpy as np

import dwarf.ready_solutions.expertise_solutions  # noqa: F401  registers metrics
from dwarf.core.expertise_orchestrator.expertise_core import Expertise_Core

IMPERCEPTIBILITY_METRICS = frozenset(
    {
        "BRISQUE",
        "DISTS",
        "FSIM",
        "FSIMc",
        "LPIPS",
        "MSE",
        "MS_SSIM",
        "NIQE",
        "PSNR",
        "SSIM",
        "VIF",
    }
)

ROBUSTNESS_METRICS = frozenset(
    {
        "AUC",
        "Accuracy",
        "BER",
        "F1",
        "NC",
        "P_Value",
        "Precision",
        "Recall",
        "Watermark_PSNR",
    }
)

EXPECTED_METRICS = IMPERCEPTIBILITY_METRICS | ROBUSTNESS_METRICS
PYIQA_METRICS = frozenset({"BRISQUE", "DISTS", "LPIPS", "NIQE"})
FULL_REFERENCE_METRICS = frozenset({"DISTS", "FSIM", "FSIMc", "LPIPS", "MSE", "MS_SSIM", "PSNR", "SSIM", "VIF"})
NO_REFERENCE_METRICS = frozenset({"BRISQUE", "NIQE"})
BIT_METRICS = frozenset({"BER", "NC"})
DETECTOR_LABEL_METRICS = frozenset({"Accuracy", "F1", "Precision", "Recall"})


def registered_metrics() -> dict:
    """Return concrete registered metrics, excluding abstract categories."""
    return {
        name: metric
        for name, metric in Expertise_Core.get_registered_expertises().items()
        if not name.startswith("Ready_")
    }


def evaluate(name: str, **arguments) -> float:
    """Evaluate a metric without suppressing implementation failures."""
    metric = Expertise_Core.get_expertise_class_by_name(name)
    return metric.expertise(**arguments)


def arguments_for(name: str, original: np.ndarray, distorted: np.ndarray) -> dict:
    """Build representative valid arguments for a registered metric."""
    if name in NO_REFERENCE_METRICS:
        return {"input_image": distorted}
    if name in FULL_REFERENCE_METRICS:
        return {"original_image": original, "distorted_image": distorted}
    if name in BIT_METRICS:
        return {"original_bits": "1011001010", "extracted_bits": "1011001011"}
    if name == "Watermark_PSNR":
        return {"original_watermark": original, "extracted_watermark": distorted}
    if name == "AUC":
        return {
            "y_true": np.array([1, 0, 1, 0]),
            "y_scores": np.array([0.9, 0.1, 0.8, 0.2]),
        }
    if name == "P_Value":
        return {"statistic": 2.0, "null_samples": np.array([0.0, 1.0, 3.0])}
    if name in DETECTOR_LABEL_METRICS:
        return {
            "y_true": np.array([1, 0, 1, 0]),
            "y_pred": np.array([1, 0, 0, 0]),
        }
    raise AssertionError(f"No representative arguments configured for {name}")


def to_luma(image: np.ndarray) -> np.ndarray:
    """Convert an RGB test image to the luma used by the implementations."""
    image = image.astype(np.float64)
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def add_noise(image: np.ndarray, sigma: float, seed: int = 5) -> np.ndarray:
    """Add deterministic Gaussian noise and return an uint8 image."""
    noise = np.random.default_rng(seed).normal(0.0, sigma, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
