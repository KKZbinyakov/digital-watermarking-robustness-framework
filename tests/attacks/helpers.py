"""Shared data and helpers for attack tests."""

import importlib.util
import shutil

import numpy as np
import pytest
from PIL import features

import dwarf.ready_solutions.attack_solutions  # noqa: F401 - imports populate the registry
from dwarf.core.attack_orchestrator.attack_core import Attack_Core

EXPECTED_ATTACKS = frozenset(
    {
        "AWGN",
        "Anisotropic_Diffusion",
        "Avif",
        "Bilateral_Filter",
        "Bit_Depth_Reduction",
        "Box_Filter",
        "Bpg",
        "Brightness_Contrast",
        "Color_Jitter",
        "Color_Quantization",
        "Color_Space_Noise",
        "Crop",
        "Dithering",
        "Flif",
        "Gamma_Correction",
        "Gaussian_Blur",
        "Grayscale",
        "Heic",
        "Histogram_Equalization",
        "Homomorphic_Filter",
        "Impulse",
        "Jpeg",
        "Jpeg2000",
        "Median_Filter",
        "Periodic",
        "Poisson",
        "Salt_and_Pepper",
        "Speckle",
        "Tiff",
        "Unsharp_Mask",
        "Webp",
        "Wiener_Filter",
    }
)

SEEDED_ATTACKS = frozenset(
    {
        "AWGN",
        "Color_Jitter",
        "Color_Space_Noise",
        "Impulse",
        "Periodic",
        "Poisson",
        "Salt_and_Pepper",
        "Speckle",
    }
)

LOSSLESS_ATTACKS = frozenset({"Flif", "Tiff"})


def concrete_attacks():
    """Return registry entries that represent concrete attacks."""
    return {
        name: attack for name, attack in Attack_Core.get_registered_attacks().items() if not name.startswith("Ready_")
    }


ATTACKS = concrete_attacks()


def unavailable_reason(name):
    """Describe a missing build artifact or optional runtime dependency."""
    if name not in ATTACKS:
        return f"{name} is absent from the registry; the registry test reports the missing build artifact"
    if name == "Avif" and not (importlib.util.find_spec("pillow_avif") or importlib.util.find_spec("pillow_heif")):
        return "AVIF support requires pillow-avif-plugin or pillow-heif"
    if name == "Heic" and importlib.util.find_spec("pillow_heif") is None:
        return "HEIC support requires pillow-heif"
    if name == "Bpg" and (shutil.which("bpgenc") is None or shutil.which("bpgdec") is None):
        return "BPG support requires both bpgenc and bpgdec"
    if name == "Flif" and shutil.which("flif") is None:
        return "FLIF support requires the flif executable"
    if name == "Bilateral_Filter" and importlib.util.find_spec("cv2") is None:
        return "Bilateral_Filter requires opencv-python-headless"
    if name == "Jpeg2000" and not features.check("jpg_2000"):
        return "the installed Pillow build has no JPEG 2000 support"
    if name == "Webp" and not features.check("webp"):
        return "the installed Pillow build has no WebP support"
    return None


def attack_cases(names=EXPECTED_ATTACKS):
    """Build parametrization cases with narrowly scoped capability skips."""
    cases = []
    for name in sorted(names):
        reason = unavailable_reason(name)
        if reason is None:
            cases.append(pytest.param(name, id=name))
        else:
            cases.append(pytest.param(name, id=name, marks=pytest.mark.skip(reason=reason)))
    return cases


def attack_class(name):
    """Return an attack class, skipping only when its extension was not built."""
    if name not in ATTACKS:
        pytest.skip(f"{name} is absent from the registry; see the registry completeness test")
    return ATTACKS[name]


def run_attack(name, image, **params):
    """Run an available attack without suppressing implementation errors."""
    reason = unavailable_reason(name)
    if reason is not None:
        pytest.skip(reason)
    return attack_class(name).attack(input_image=image, **params)


def mean_absolute_error(image, reference):
    """Return the mean absolute per-channel pixel error."""
    return float(np.abs(image.astype(np.float64) - reference.astype(np.float64)).mean())


def changed_pixel_count(image, reference):
    """Count pixels that differ in at least one channel."""
    return int(np.count_nonzero(np.any(image != reference, axis=2)))


def high_frequency_energy(image):
    """Estimate fine detail through adjacent-pixel differences."""
    data = image.astype(np.float64)
    return float(np.abs(np.diff(data, axis=1)).mean() + np.abs(np.diff(data, axis=0)).mean())
