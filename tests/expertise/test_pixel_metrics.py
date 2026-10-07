"""Tests for MSE, PSNR, SSIM, and MS-SSIM."""

import numpy as np
import pytest
from sewar.full_ref import msssim
from skimage.metrics import structural_similarity

from .helpers import add_noise, evaluate, to_luma

IDENTICAL_LIMITS = {
    "MSE": 0.0,
    "MS_SSIM": 1.0,
    "PSNR": float("inf"),
    "SSIM": 1.0,
}


@pytest.mark.parametrize("name, expected", sorted(IDENTICAL_LIMITS.items()))
def test_identical_images_reach_metric_limit(name, expected, photo):
    value = evaluate(name, original_image=photo, distorted_image=photo)

    if np.isinf(expected):
        assert value == expected
    else:
        assert value == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("offset", [1, 4, 16, 64])
def test_mse_matches_closed_form(offset, photo):
    shifted = np.clip(photo.astype(np.int16) - offset, 0, 255).astype(np.uint8)
    difference = photo.astype(np.float64) - shifted.astype(np.float64)

    value = evaluate("MSE", original_image=photo, distorted_image=shifted)

    assert value == pytest.approx(float(np.mean(difference**2)))


@pytest.mark.parametrize("offset", [1, 4, 16, 64])
def test_psnr_matches_closed_form(offset, photo):
    shifted = np.clip(photo.astype(np.int16) - offset, 0, 255).astype(np.uint8)
    difference = photo.astype(np.float64) - shifted.astype(np.float64)
    expected = 10 * np.log10(255**2 / np.mean(difference**2))

    value = evaluate("PSNR", original_image=photo, distorted_image=shifted)

    assert value == pytest.approx(expected)


def test_psnr_and_mse_use_the_same_error(photo, distorted):
    mse = evaluate("MSE", original_image=photo, distorted_image=distorted)
    psnr = evaluate("PSNR", original_image=photo, distorted_image=distorted)

    assert psnr == pytest.approx(10 * np.log10(255**2 / mse))


def test_ssim_matches_scikit_image(photo, distorted):
    expected = structural_similarity(
        to_luma(photo),
        to_luma(distorted),
        data_range=255,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )

    value = evaluate("SSIM", original_image=photo, distorted_image=distorted)

    assert value == pytest.approx(expected)


def test_ms_ssim_matches_sewar(photo, distorted):
    expected = float(msssim(to_luma(photo), to_luma(distorted), MAX=255))

    value = evaluate("MS_SSIM", original_image=photo, distorted_image=distorted)

    assert value == pytest.approx(expected)


def test_mse_grows_with_noise_strength(photo):
    values = [
        evaluate("MSE", original_image=photo, distorted_image=add_noise(photo, sigma))
        for sigma in (2.0, 8.0, 20.0, 40.0)
    ]

    assert all(values[index] < values[index + 1] for index in range(len(values) - 1))


@pytest.mark.parametrize("name", ["MS_SSIM", "PSNR", "SSIM"])
def test_similarity_decreases_with_noise_strength(name, photo):
    values = [
        evaluate(name, original_image=photo, distorted_image=add_noise(photo, sigma))
        for sigma in (2.0, 8.0, 20.0, 40.0)
    ]

    assert all(values[index] > values[index + 1] for index in range(len(values) - 1)), values


@pytest.mark.parametrize("name", ["MSE", "PSNR"])
def test_rgb_metrics_reject_single_channel_input(name, photo):
    with pytest.raises(ValueError):
        evaluate(name, original_image=to_luma(photo), distorted_image=photo)


@pytest.mark.parametrize("name", ["MS_SSIM", "SSIM"])
def test_luma_metrics_accept_single_channel_input(name, photo):
    luma = to_luma(photo)

    assert evaluate(name, original_image=luma, distorted_image=luma) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name", ["MS_SSIM", "SSIM"])
def test_luma_and_rgb_inputs_agree(name, photo, distorted):
    from_rgb = evaluate(name, original_image=photo, distorted_image=distorted)
    from_luma = evaluate(name, original_image=to_luma(photo), distorted_image=to_luma(distorted))

    assert from_rgb == pytest.approx(from_luma)


@pytest.mark.parametrize("name", ["MSE", "MS_SSIM", "PSNR", "SSIM"])
def test_rejects_malformed_image_matrix(name, photo):
    with pytest.raises(ValueError):
        evaluate(name, original_image=np.zeros((4, 4, 4, 4)), distorted_image=photo)


@pytest.mark.parametrize("name", ["MSE", "MS_SSIM", "PSNR", "SSIM"])
def test_rejects_different_image_shapes(name, photo):
    with pytest.raises(ValueError):
        evaluate(name, original_image=photo, distorted_image=photo[:-1])


def test_ssim_rejects_image_smaller_than_window(photo):
    tiny = photo[:10, :10]

    with pytest.raises(ValueError, match="smaller than the window"):
        evaluate("SSIM", original_image=tiny, distorted_image=tiny)


def test_ms_ssim_rejects_image_smaller_than_five_scale_minimum(photo):
    tiny = photo[:175, :175]

    with pytest.raises(ValueError, match="at least 176"):
        evaluate("MS_SSIM", original_image=tiny, distorted_image=tiny)
