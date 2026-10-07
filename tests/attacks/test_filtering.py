"""Tests for spatial and frequency-domain filtering attacks."""

import sys

import numpy as np
import pytest
from PIL import Image

from .helpers import attack_class, high_frequency_energy, mean_absolute_error, run_attack


def test_stronger_gaussian_blur_removes_more_high_frequency_energy(photo):
    weak_energy = high_frequency_energy(run_attack("Gaussian_Blur", photo, sigma=0.5))
    strong_energy = high_frequency_energy(run_attack("Gaussian_Blur", photo, sigma=5.0))
    assert strong_energy < weak_energy


def test_larger_box_filter_removes_more_high_frequency_energy(photo):
    small_energy = high_frequency_energy(run_attack("Box_Filter", photo, window=3))
    large_energy = high_frequency_energy(run_attack("Box_Filter", photo, window=15))
    assert large_energy < small_energy


@pytest.mark.parametrize("option", [1, 2])
def test_more_diffusion_iterations_smooth_more_for_each_conductance(option, photo):
    few_energy = high_frequency_energy(run_attack("Anisotropic_Diffusion", photo, iterations=1, option=option))
    many_energy = high_frequency_energy(run_attack("Anisotropic_Diffusion", photo, iterations=50, option=option))
    assert many_energy < few_energy


def test_unsharp_mask_increases_high_frequency_energy(photo):
    assert high_frequency_energy(run_attack("Unsharp_Mask", photo)) > high_frequency_energy(photo)


def test_stronger_unsharp_amount_increases_high_frequency_energy(photo):
    weak_energy = high_frequency_energy(run_attack("Unsharp_Mask", photo, amount=0.2))
    strong_energy = high_frequency_energy(run_attack("Unsharp_Mask", photo, amount=4.0))
    assert strong_energy > weak_energy


def test_median_filter_reduces_salt_and_pepper_noise(photo):
    noisy = run_attack("Salt_and_Pepper", photo, density=0.05, seed=1)
    cleaned = run_attack("Median_Filter", noisy, window=3)
    assert mean_absolute_error(cleaned, photo) < mean_absolute_error(noisy, photo)


def test_wiener_filter_reduces_gaussian_noise(photo):
    noisy = run_attack("AWGN", photo, sigma=0.05, seed=1)
    cleaned = run_attack("Wiener_Filter", noisy)
    assert mean_absolute_error(cleaned, photo) < mean_absolute_error(noisy, photo)


def test_bilateral_filter_reduces_gaussian_noise(photo):
    noisy = run_attack("AWGN", photo, sigma=0.05, seed=1)
    cleaned = run_attack("Bilateral_Filter", noisy)
    assert mean_absolute_error(cleaned, photo) < mean_absolute_error(noisy, photo)


def test_bilateral_filter_reports_missing_opencv_dependency(monkeypatch, small_photo):
    monkeypatch.setitem(sys.modules, "cv2", None)

    with pytest.raises(RuntimeError, match="opencv-python-headless"):
        attack_class("Bilateral_Filter").attack(input_image=small_photo)


def test_homomorphic_filter_changes_luma_but_preserves_chroma(photo):
    result = run_attack("Homomorphic_Filter", photo)
    original_ycbcr = np.asarray(Image.fromarray(photo, "RGB").convert("YCbCr"), dtype=np.float64)
    result_ycbcr = np.asarray(Image.fromarray(result, "RGB").convert("YCbCr"), dtype=np.float64)

    assert mean_absolute_error(result_ycbcr[..., 1:], original_ycbcr[..., 1:]) < 2.0
    assert not np.array_equal(result_ycbcr[..., 0], original_ycbcr[..., 0])


def test_higher_homomorphic_high_gain_boosts_detail(photo):
    mild_energy = high_frequency_energy(run_attack("Homomorphic_Filter", photo, gamma_high=1.2))
    strong_energy = high_frequency_energy(run_attack("Homomorphic_Filter", photo, gamma_high=3.0))
    assert strong_energy > mild_energy


@pytest.mark.parametrize("level", [0, 64, 128, 200, 255])
def test_homomorphic_filter_keeps_uniform_frame_at_same_level(level):
    uniform = np.full((32, 32, 3), level, dtype=np.uint8)
    result = run_attack("Homomorphic_Filter", uniform)
    assert mean_absolute_error(result, uniform) < 1.5


@pytest.mark.parametrize("spike", [130, 180, 255])
def test_homomorphic_normalization_is_stable_for_single_outlier(spike, photo):
    spiked = photo.copy()
    spiked[0, 0] = spike
    baseline = run_attack("Homomorphic_Filter", photo).astype(np.float64)
    result = run_attack("Homomorphic_Filter", spiked).astype(np.float64)

    assert abs(result.mean() - baseline.mean()) < 0.5


def test_lower_homomorphic_cutoff_distorts_more(photo):
    weak_error = mean_absolute_error(run_attack("Homomorphic_Filter", photo, cutoff=100.0), photo)
    strong_error = mean_absolute_error(run_attack("Homomorphic_Filter", photo, cutoff=10.0), photo)
    assert strong_error > weak_error


@pytest.mark.parametrize(
    ("name", "params"),
    [
        pytest.param("Anisotropic_Diffusion", {"iterations": 0}, id="diffusion-iterations-below"),
        pytest.param("Anisotropic_Diffusion", {"iterations": 51}, id="diffusion-iterations-above"),
        pytest.param("Anisotropic_Diffusion", {"kappa": 0.0}, id="diffusion-kappa-below"),
        pytest.param("Anisotropic_Diffusion", {"kappa": 0.51}, id="diffusion-kappa-above"),
        pytest.param("Anisotropic_Diffusion", {"gamma": 0.0}, id="diffusion-gamma-below"),
        pytest.param("Anisotropic_Diffusion", {"gamma": 0.251}, id="diffusion-gamma-above"),
        pytest.param("Anisotropic_Diffusion", {"option": 3}, id="diffusion-option"),
        pytest.param("Bilateral_Filter", {"sigma_color": 0.0}, id="bilateral-color-below"),
        pytest.param("Bilateral_Filter", {"sigma_color": 0.51}, id="bilateral-color-above"),
        pytest.param("Bilateral_Filter", {"sigma_space": 0.0}, id="bilateral-space-below"),
        pytest.param("Bilateral_Filter", {"sigma_space": 10.1}, id="bilateral-space-above"),
        pytest.param("Box_Filter", {"window": 2}, id="box-too-small"),
        pytest.param("Box_Filter", {"window": 4}, id="box-even"),
        pytest.param("Gaussian_Blur", {"sigma": 0.49}, id="gaussian-below"),
        pytest.param("Gaussian_Blur", {"sigma": 5.01}, id="gaussian-above"),
        pytest.param("Homomorphic_Filter", {"gamma_low": 0.09}, id="homomorphic-low-gain-below"),
        pytest.param("Homomorphic_Filter", {"gamma_low": 1.01}, id="homomorphic-low-gain-above"),
        pytest.param("Homomorphic_Filter", {"gamma_high": 0.99}, id="homomorphic-high-gain-below"),
        pytest.param("Homomorphic_Filter", {"gamma_high": 3.01}, id="homomorphic-high-gain-above"),
        pytest.param("Homomorphic_Filter", {"cutoff": 9.99}, id="homomorphic-cutoff-below"),
        pytest.param("Homomorphic_Filter", {"cutoff": 100.1}, id="homomorphic-cutoff-above"),
        pytest.param("Homomorphic_Filter", {"c": 0.0}, id="homomorphic-steepness-below"),
        pytest.param("Homomorphic_Filter", {"c": 10.01}, id="homomorphic-steepness-above"),
        pytest.param("Median_Filter", {"window": 1}, id="median-too-small"),
        pytest.param("Median_Filter", {"window": 4}, id="median-even"),
        pytest.param("Median_Filter", {"window": 9}, id="median-too-large"),
        pytest.param("Unsharp_Mask", {"amount": 0.0}, id="unsharp-amount-below"),
        pytest.param("Unsharp_Mask", {"amount": 5.1}, id="unsharp-amount-above"),
        pytest.param("Unsharp_Mask", {"radius": 0.49}, id="unsharp-radius-below"),
        pytest.param("Unsharp_Mask", {"radius": 5.01}, id="unsharp-radius-above"),
        pytest.param("Unsharp_Mask", {"threshold": -0.01}, id="unsharp-threshold-below"),
        pytest.param("Unsharp_Mask", {"threshold": 1.01}, id="unsharp-threshold-above"),
        pytest.param("Wiener_Filter", {"window": 2}, id="wiener-too-small"),
        pytest.param("Wiener_Filter", {"window": 4}, id="wiener-even"),
        pytest.param("Wiener_Filter", {"noise": -0.01}, id="wiener-noise"),
    ],
)
def test_filter_parameters_outside_documented_range_raise(name, params, small_photo):
    with pytest.raises(ValueError):
        attack_class(name).attack(input_image=small_photo, **params)
