"""Tests for colour and brightness attacks."""

import cv2
import numpy as np
import pytest
from PIL import Image

from .helpers import attack_class, run_attack


def test_grayscale_produces_equal_rgb_channels(photo):
    result = run_attack("Grayscale", photo)
    assert np.array_equal(result[..., 0], result[..., 1])
    assert np.array_equal(result[..., 1], result[..., 2])


def test_gamma_one_is_bit_exact_identity(photo):
    assert np.array_equal(run_attack("Gamma_Correction", photo, gamma=1.0), photo)


def test_neutral_brightness_and_contrast_are_bit_exact_identity(photo):
    assert np.array_equal(
        run_attack("Brightness_Contrast", photo, brightness=1.0, contrast=1.0),
        photo,
    )


def test_zero_color_jitter_is_identity_with_rgb_hsv_round_trip_tolerance(photo):
    result = run_attack(
        "Color_Jitter",
        photo,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
        seed=1,
    )

    assert np.abs(result.astype(np.int16) - photo.astype(np.int16)).max() <= 1


@pytest.mark.parametrize(
    ("space", "round_trip_tolerance"),
    [
        pytest.param("YCbCr", 4, id="ycbcr"),
        pytest.param("HSV", 3, id="hsv"),
        pytest.param("LAB", 8, id="lab"),
    ],
)
def test_zero_color_space_noise_is_only_the_documented_round_trip(space, round_trip_tolerance, small_photo):
    result = run_attack("Color_Space_Noise", small_photo, space=space, noise_std=0.0, seed=1)
    expected = np.asarray(Image.fromarray(small_photo, "RGB").convert(space).convert("RGB"))

    assert np.array_equal(result, expected)
    assert np.abs(result.astype(np.int16) - small_photo.astype(np.int16)).max() <= round_trip_tolerance


@pytest.mark.parametrize("space", ["YCbCr", "HSV", "LAB"])
def test_documented_color_space_noise_is_seeded_and_nonzero(space, small_photo):
    baseline = run_attack("Color_Space_Noise", small_photo, space=space, noise_std=0.0, seed=7)
    first = run_attack("Color_Space_Noise", small_photo, space=space, noise_std=10.0, seed=7)
    second = run_attack("Color_Space_Noise", small_photo, space=space, noise_std=10.0, seed=7)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, baseline)


@pytest.mark.parametrize(("gamma", "brighter"), [(0.5, True), (2.0, False)])
def test_gamma_moves_mean_brightness_in_expected_direction(gamma, brighter, photo):
    result_mean = run_attack("Gamma_Correction", photo, gamma=gamma).astype(np.float64).mean()
    source_mean = photo.astype(np.float64).mean()
    assert bool(result_mean > source_mean) is brighter


def test_brightness_factor_moves_mean_brightness(photo):
    darker = run_attack("Brightness_Contrast", photo, brightness=0.6, contrast=1.0)
    brighter = run_attack("Brightness_Contrast", photo, brightness=1.4, contrast=1.0)

    assert darker.mean() < photo.mean() < brighter.mean()


def test_eight_bit_reduction_is_identity(photo):
    assert np.array_equal(run_attack("Bit_Depth_Reduction", photo, bits=8), photo)


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 7])
def test_bit_depth_reduction_uses_only_declared_levels(bits, photo):
    result = run_attack("Bit_Depth_Reduction", photo, bits=bits)
    expected_levels = np.round(np.arange(1 << bits) / ((1 << bits) - 1) * 255).astype(np.uint8)

    assert set(np.unique(result)) <= set(expected_levels)


@pytest.mark.parametrize("colors", [2, 4, 8, 16])
@pytest.mark.parametrize("method", ["median_cut", "kmeans"])
def test_quantization_limits_palette_size(colors, method, photo):
    result = run_attack("Color_Quantization", photo, colors=colors, method=method, seed=1)
    assert len(np.unique(result.reshape(-1, 3), axis=0)) == colors


@pytest.mark.parametrize("levels", [2, 3, 4])
@pytest.mark.parametrize("method", ["floyd_steinberg", "ordered"])
def test_dithering_uses_requested_channel_level_count(levels, method, photo):
    result = run_attack("Dithering", photo, levels=levels, method=method)
    expected_levels = set(np.round(np.linspace(0, 255, levels)).astype(np.uint8).tolist())
    assert set(np.unique(result).tolist()) == expected_levels


@pytest.mark.parametrize("method", ["floyd_steinberg", "ordered"])
def test_dithering_approximately_preserves_mean_brightness(method, photo):
    result = run_attack("Dithering", photo, levels=2, method=method)
    assert abs(result.astype(np.float64).mean() - photo.astype(np.float64).mean()) < 12


@pytest.mark.parametrize("method", ["global", "clahe"])
def test_histogram_equalization_preserves_or_widens_dynamic_range(method, photo):
    result = run_attack("Histogram_Equalization", photo, method=method)
    assert result.astype(np.float64).std() >= photo.astype(np.float64).std() * 0.9


def test_clahe_matches_opencv_luma_reference(small_photo):
    source_ycbcr = np.asarray(Image.fromarray(small_photo, "RGB").convert("YCbCr"))
    reference_luma = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(source_ycbcr[..., 0])

    result = run_attack(
        "Histogram_Equalization",
        small_photo,
        method="clahe",
        tiles=8,
        clip_limit=2.0,
        bins=256,
    )
    result_luma = np.asarray(Image.fromarray(result, "RGB").convert("YCbCr"))[..., 0]
    error = np.abs(result_luma.astype(np.int16) - reference_luma.astype(np.int16))

    assert error.mean() < 3.0
    assert error.max() <= 12


@pytest.mark.xfail(
    strict=True,
    reason="ordered dithering documents a power-of-two matrix but currently accepts matrix_size=3",
)
def test_ordered_dithering_rejects_non_power_of_two_matrix(small_photo):
    with pytest.raises(ValueError):
        attack_class("Dithering").attack(input_image=small_photo, method="ordered", matrix_size=3)


@pytest.mark.parametrize(
    ("name", "params"),
    [
        pytest.param("Gamma_Correction", {"gamma": 0}, id="gamma-zero"),
        pytest.param("Gamma_Correction", {"gamma": -1}, id="gamma-negative"),
        pytest.param("Bit_Depth_Reduction", {"bits": 0}, id="bits-zero"),
        pytest.param("Bit_Depth_Reduction", {"bits": 9}, id="bits-nine"),
        pytest.param("Color_Quantization", {"colors": 1}, id="too-few-colors"),
        pytest.param("Color_Quantization", {"method": "kohonen"}, id="unknown-quantizer"),
        pytest.param(
            "Color_Quantization",
            {"method": "kmeans", "colors": 101, "sample_size": 100},
            id="palette-exceeds-sample",
        ),
        pytest.param("Dithering", {"levels": 1}, id="too-few-dither-levels"),
        pytest.param("Dithering", {"method": "ordered", "matrix_size": 1}, id="small-bayer-matrix"),
        pytest.param("Dithering", {"method": "bayer"}, id="unknown-dithering"),
        pytest.param("Histogram_Equalization", {"method": "adaptive"}, id="unknown-equalizer"),
        pytest.param("Histogram_Equalization", {"method": "clahe", "tiles": 1}, id="too-few-tiles"),
        pytest.param("Histogram_Equalization", {"method": "clahe", "bins": 1}, id="too-few-bins"),
        pytest.param("Histogram_Equalization", {"method": "clahe", "bins": 257}, id="too-many-bins"),
    ],
)
def test_invalid_color_attack_parameters_raise_value_error(name, params, small_photo):
    with pytest.raises(ValueError):
        attack_class(name).attack(input_image=small_photo, **params)
