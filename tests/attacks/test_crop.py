"""Tests specific to the geometric crop attack."""

import numpy as np
import pytest

from .helpers import attack_class, run_attack


@pytest.mark.parametrize("mode", ["pad", "resize", "raw"])
def test_ratio_one_is_identity_for_every_crop_mode(mode, photo):
    assert np.array_equal(run_attack("Crop", photo, ratio=1.0, mode=mode), photo)


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(0.5, (32, 48, 3)), (0.25, (16, 24, 3)), (1.0, (64, 96, 3))],
)
def test_raw_crop_size_follows_each_input_dimension(ratio, expected, small_photo):
    assert run_attack("Crop", small_photo, mode="raw", ratio=ratio).shape == expected


@pytest.mark.parametrize("mode", ["pad", "resize"])
def test_restoring_crop_modes_keep_original_size(mode, small_photo):
    assert run_attack("Crop", small_photo, mode=mode).shape == small_photo.shape


@pytest.mark.parametrize("resample", ["nearest", "bilinear", "bicubic", "lanczos"])
def test_resize_supports_every_documented_resampler(resample, small_photo):
    result = run_attack("Crop", small_photo, mode="resize", resample=resample)
    assert result.shape == small_photo.shape


@pytest.mark.parametrize(
    ("position", "corner"),
    [
        ("top_left", (0, 0)),
        ("top_right", (0, 48)),
        ("bottom_left", (32, 0)),
        ("bottom_right", (32, 48)),
        ("center", (16, 24)),
    ],
)
def test_pad_mode_places_kept_region_at_requested_anchor(position, corner, small_photo):
    fill = (255, 0, 255)
    result = run_attack("Crop", small_photo, position=position, fill=fill)
    expected = np.empty_like(small_photo)
    expected[:] = fill
    top, left = corner
    kept_height = small_photo.shape[0] // 2
    kept_width = small_photo.shape[1] // 2
    expected[top : top + kept_height, left : left + kept_width] = small_photo[
        top : top + kept_height,
        left : left + kept_width,
    ]

    assert np.array_equal(result, expected)


def test_pad_mode_uses_rgb_fill_for_discarded_area(small_photo):
    result = run_attack("Crop", small_photo, position="top_left", fill=(10, 20, 30))
    assert tuple(result[-1, -1]) == (10, 20, 30)


def test_scalar_fill_is_applied_to_all_channels(small_photo):
    result = run_attack("Crop", small_photo, position="top_left", fill=17)
    assert tuple(result[-1, -1]) == (17, 17, 17)


def test_random_position_is_reproducible_and_changes_with_seed(small_photo):
    first = run_attack("Crop", small_photo, position="random", mode="raw", seed=4)
    second = run_attack("Crop", small_photo, position="random", mode="raw", seed=4)
    different_seed = run_attack("Crop", small_photo, position="random", mode="raw", seed=5)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, different_seed)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"ratio": 0}, id="zero-ratio"),
        pytest.param({"ratio": 1.5}, id="ratio-above-one"),
        pytest.param({"position": "middle"}, id="unknown-position"),
        pytest.param({"mode": "stretch"}, id="unknown-mode"),
        pytest.param({"mode": "resize", "resample": "spline"}, id="unknown-resampler"),
        pytest.param({"fill": (1, 2)}, id="short-fill"),
        pytest.param({"fill": (1, 2, 3, 4)}, id="long-fill"),
        pytest.param({"fill": -1}, id="negative-fill"),
        pytest.param({"fill": 256}, id="fill-above-255"),
    ],
)
def test_invalid_crop_parameters_raise_value_error(params, small_photo):
    with pytest.raises(ValueError):
        attack_class("Crop").attack(input_image=small_photo, **params)
