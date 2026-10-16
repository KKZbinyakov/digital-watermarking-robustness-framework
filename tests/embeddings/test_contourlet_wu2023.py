"""Regression tests for the Wu et al. 2023 contourlet paper profile."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dwarf.ready_solutions.embedding_solutions.frequency.contourlet_wu2023 import (
    Contourlet_Wu2023,
    _adaptive_steps,
    _arnold,
    _lowpass_analysis,
    _lowpass_synthesis_delta,
)

ROOT = Path(__file__).resolve().parents[2]


def _host() -> np.ndarray:
    image = Image.open(ROOT / "Asuka.jpg").convert("RGB")
    return np.array(image.resize((512, 512), Image.Resampling.LANCZOS))


def _nc(original: np.ndarray, extracted: np.ndarray) -> float:
    denominator = np.sqrt(np.sum(original * original) * np.sum(extracted * extracted))
    return float(np.sum(original * extracted) / denominator)


def test_adaptive_step_uses_published_energy_formula():
    block = np.arange(1.0, 17.0).reshape(1, 4, 4)
    energy = np.mean(block**2)

    actual = _adaptive_steps(block, 48.0)[0]

    assert actual == pytest.approx(np.log2((energy + 1000.0) / 1000.0) * 48.0)


def test_pkva_lowpass_delta_is_recovered_by_analysis():
    delta = np.random.default_rng(7).normal(size=(32, 32))

    recovered = _lowpass_analysis(_lowpass_synthesis_delta(delta))

    np.testing.assert_allclose(recovered, delta, rtol=1e-12, atol=1e-12)


def test_arnold_scrambling_round_trip():
    bits = np.random.default_rng(3).integers(0, 2, 32 * 32, dtype=np.uint8)

    scrambled = _arnold(bits, 11)
    restored = _arnold(scrambled, 11, inverse=True)

    assert not np.array_equal(scrambled, bits)
    np.testing.assert_array_equal(restored, bits)


def test_paper_profile_has_lossless_clean_extraction_and_reported_psnr_order():
    image = _host()
    bits = np.random.default_rng(42).integers(0, 2, 32 * 32, dtype=np.uint8)

    marked = Contourlet_Wu2023.embedding(
        input_image=image,
        watermark_bits=bits,
        arnold_iterations=5,
        original_image=image,
    )
    extracted = Contourlet_Wu2023.extraction(
        input_image=marked,
        num_bits=bits.size,
        arnold_iterations=5,
        attacked_image=marked,
    )

    np.testing.assert_array_equal(extracted, bits)
    mse = np.mean((image.astype(np.float64) - marked.astype(np.float64)) ** 2)
    psnr = 10.0 * np.log10(255.0**2 / mse)
    assert 40.0 <= psnr <= 48.0


@pytest.mark.parametrize(
    "quality, minimum_nc",
    [
        (90, 0.96),
        (70, 0.96),
        (50, 0.94),
        (30, 0.84),
        (10, 0.75),
    ],
)
def test_jpeg_nc_is_in_the_range_reported_by_the_paper(quality, minimum_nc):
    image = _host()
    bits = np.random.default_rng(42).integers(0, 2, 32 * 32, dtype=np.uint8)
    marked = Contourlet_Wu2023.embedding(input_image=image, watermark_bits=bits, arnold_iterations=5)
    buffer = io.BytesIO()
    Image.fromarray(marked).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    attacked = np.array(Image.open(buffer).convert("RGB"))

    extracted = Contourlet_Wu2023.extraction(input_image=attacked, num_bits=bits.size, arnold_iterations=5)

    assert minimum_nc <= _nc(bits, extracted) <= 1.0


def test_rotation_normalization_reaches_the_paper_nc_range():
    image = _host()
    bits = np.random.default_rng(42).integers(0, 2, 32 * 32, dtype=np.uint8)
    marked = Contourlet_Wu2023.embedding(input_image=image, watermark_bits=bits, arnold_iterations=5)
    rotated = Image.fromarray(marked).rotate(
        30,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0),
    )

    extracted = Contourlet_Wu2023.extraction(
        input_image=np.array(rotated),
        num_bits=bits.size,
        arnold_iterations=5,
    )

    assert _nc(bits, extracted) >= 0.82
