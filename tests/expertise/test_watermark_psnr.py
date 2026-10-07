"""Tests for PSNR of image watermarks."""

import numpy as np
import pytest

from .helpers import evaluate


def test_identical_watermarks_have_infinite_psnr(photo):
    value = evaluate("Watermark_PSNR", original_watermark=photo, extracted_watermark=photo)

    assert value == float("inf")


def test_watermark_psnr_matches_closed_form():
    original = np.full((24, 32), 100, dtype=np.uint8)
    extracted = np.full((24, 32), 110, dtype=np.uint8)
    expected = 10 * np.log10(255**2 / 100)

    value = evaluate(
        "Watermark_PSNR",
        original_watermark=original,
        extracted_watermark=extracted,
    )

    assert value == pytest.approx(expected)


def test_single_channel_and_repeated_rgb_layouts_agree():
    original = (np.random.default_rng(3).random((32, 32)) > 0.5).astype(np.uint8) * 255
    extracted = np.clip(original.astype(np.float64) + 20, 0, 255).astype(np.uint8)

    single_channel = evaluate(
        "Watermark_PSNR",
        original_watermark=original,
        extracted_watermark=extracted,
    )
    repeated_rgb = evaluate(
        "Watermark_PSNR",
        original_watermark=np.stack([original] * 3, axis=-1),
        extracted_watermark=np.stack([extracted] * 3, axis=-1),
    )

    assert single_channel == pytest.approx(repeated_rgb)


def test_different_sizes_are_compared_on_common_crop():
    original = np.arange(8 * 10, dtype=np.float64).reshape(8, 10)
    extracted = original[:5, :7].copy()

    value = evaluate(
        "Watermark_PSNR",
        original_watermark=original,
        extracted_watermark=extracted,
    )

    assert value == float("inf")


@pytest.mark.parametrize("argument", ["original_watermark", "extracted_watermark"])
def test_rejects_malformed_watermark_matrix(argument):
    arguments = {
        "original_watermark": np.zeros((8, 8)),
        "extracted_watermark": np.zeros((8, 8)),
    }
    arguments[argument] = np.zeros((2, 2, 2, 2))

    with pytest.raises(ValueError):
        evaluate("Watermark_PSNR", **arguments)
