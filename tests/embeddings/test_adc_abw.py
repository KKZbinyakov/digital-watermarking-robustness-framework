"""Regression tests for Guo et al.'s 2025 ADC-ABW4 profile."""

import io
from pathlib import Path

import numpy as np
from conftest import make_photo
from PIL import Image

import dwarf.ready_solutions.embedding_solutions  # noqa: F401
from dwarf import Embedding_Core
from dwarf.ready_solutions.embedding_solutions.spread_spectrum.adc_abw import (
    ADC_ABW,
    _dc_matrix,
)


def _embed(image, bits, **parameters):
    return ADC_ABW.embedding(input_image=image, watermark_bits=bits, **parameters)


def _extract(image, count, **parameters):
    return ADC_ABW.extraction(input_image=image, num_bits=count, **parameters)


def _jpeg(image, quality):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))


def _ber(expected, actual):
    return float(np.mean(expected != actual.astype(np.uint8)))


def _psnr(original, distorted):
    difference = original.astype(np.float64) - distorted.astype(np.float64)
    mse = np.mean(difference**2)
    return float(10.0 * np.log10(255.0**2 / mse))


def test_dc_matrix_uses_one_block_size_for_nondivisible_images():
    """Different block areas must not become false high-frequency content."""
    channel = np.full((131, 139), 10.0)

    dc, row_edges, column_edges, areas = _dc_matrix(channel, 64)

    np.testing.assert_array_equal(np.diff(row_edges), np.full(64, 2))
    np.testing.assert_array_equal(np.diff(column_edges), np.full(64, 2))
    np.testing.assert_array_equal(areas, np.full((64, 64), 4))
    np.testing.assert_allclose(dc, 20.0)


def test_nondivisible_trailing_strips_are_preserved():
    image = make_photo(131, 139, seed=30)
    bits = np.random.default_rng(31).integers(0, 2, 24, dtype=np.uint8)

    watermarked = _embed(image, bits, key=102, spreading_length=128, num_blocks=64)

    np.testing.assert_array_equal(watermarked[128:, :, :], image[128:, :, :])
    np.testing.assert_array_equal(watermarked[:, 128:, :], image[:, 128:, :])
    np.testing.assert_array_equal(
        _extract(watermarked, bits.size, key=102, spreading_length=128, num_blocks=64),
        bits,
    )


def test_contract_registration_determinism_and_clean_round_trip():
    image = make_photo(256, 256, seed=13)
    bits = np.random.default_rng(19).integers(0, 2, 64, dtype=np.uint8)
    before = image.copy()

    first = _embed(image, bits, key=2026)
    second = _embed(image, bits, key=2026)

    assert Embedding_Core.get_embedding_class_by_name("ADC_ABW") is ADC_ABW
    assert first.dtype == np.uint8
    assert first.flags.c_contiguous
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(_extract(first, bits.size, key=2026), bits)


def test_published_adc_abw4_profile_matches_article_scale():
    """A real photograph gives the paper's approximately 48 dB quality scale.

    This is a fast deterministic regression, not a replacement for the paper's
    undisclosed random selection of 100 UW/SIPI images and MATLAB JPEG codec.
    """
    image_path = Path(__file__).parents[2] / "Asuka.jpg"
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    bits = np.random.default_rng(3743139).integers(0, 2, 64, dtype=np.uint8)

    watermarked = _embed(image, bits)
    clean = _extract(watermarked, bits.size)
    jpeg_90 = _extract(_jpeg(watermarked, 90), bits.size)
    jpeg_50 = _extract(_jpeg(watermarked, 50), bits.size)

    # Table 2 reports 48.1 dB for ADC-ABW4. Image content changes the
    # image-adaptive ABW, so the one-image check intentionally uses a tolerance.
    assert abs(_psnr(image, watermarked) - 48.1) <= 2.25
    assert _ber(bits, clean) == 0.0
    assert _ber(bits, jpeg_90) <= 1 / 64
    assert _ber(bits, jpeg_50) <= 1 / 64


def test_smaller_attack_angle_increases_distortion():
    image = make_photo(256, 256, seed=32)
    bits = np.random.default_rng(33).integers(0, 2, 32, dtype=np.uint8)

    robust = _embed(image, bits, key=103, attack_angle=12.0)
    subtle = _embed(image, bits, key=103, attack_angle=45.0)

    assert _psnr(image, robust) < _psnr(image, subtle)
    np.testing.assert_array_equal(_extract(robust, bits.size, key=103), bits)
    np.testing.assert_array_equal(_extract(subtle, bits.size, key=103), bits)
