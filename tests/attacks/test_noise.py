"""Tests for stochastic noise attacks."""

import numpy as np
import pytest

from .helpers import attack_class, changed_pixel_count, mean_absolute_error, run_attack


def test_awgn_higher_sigma_distorts_more(photo):
    weak_error = mean_absolute_error(run_attack("AWGN", photo, sigma=0.01, seed=1), photo)
    strong_error = mean_absolute_error(run_attack("AWGN", photo, sigma=0.08, seed=1), photo)
    assert strong_error > weak_error


def test_impulse_higher_density_corrupts_more_pixels(photo):
    weak_count = changed_pixel_count(run_attack("Impulse", photo, density=0.001, seed=1), photo)
    strong_count = changed_pixel_count(run_attack("Impulse", photo, density=0.05, seed=1), photo)
    assert strong_count > weak_count


def test_periodic_higher_amplitude_distorts_more(photo):
    weak_error = mean_absolute_error(run_attack("Periodic", photo, amplitude=0.01, seed=1), photo)
    strong_error = mean_absolute_error(run_attack("Periodic", photo, amplitude=0.2, seed=1), photo)
    assert strong_error > weak_error


def test_poisson_lower_peak_distorts_more(photo):
    low_peak_error = mean_absolute_error(run_attack("Poisson", photo, peak=2.0, seed=1), photo)
    high_peak_error = mean_absolute_error(run_attack("Poisson", photo, peak=500.0, seed=1), photo)
    assert low_peak_error > high_peak_error


def test_salt_and_pepper_higher_density_corrupts_more_pixels(photo):
    weak_count = changed_pixel_count(run_attack("Salt_and_Pepper", photo, density=0.001, seed=1), photo)
    strong_count = changed_pixel_count(run_attack("Salt_and_Pepper", photo, density=0.05, seed=1), photo)
    assert strong_count > weak_count


def test_salt_and_pepper_changes_pixels_only_to_black_or_white(photo):
    result = run_attack("Salt_and_Pepper", photo, density=0.05, seed=1)
    changed_pixels = result[np.any(result != photo, axis=2)]
    is_salt = np.all(changed_pixels == 255, axis=1)
    is_pepper = np.all(changed_pixels == 0, axis=1)

    assert changed_pixels.size > 0
    assert np.all(is_salt | is_pepper)


def test_speckle_higher_variance_distorts_more(photo):
    weak_error = mean_absolute_error(run_attack("Speckle", photo, variance=0.001, seed=1), photo)
    strong_error = mean_absolute_error(run_attack("Speckle", photo, variance=0.05, seed=1), photo)
    assert strong_error > weak_error


def test_speckle_keeps_black_pixels_black():
    black = np.zeros((16, 24, 3), dtype=np.uint8)
    assert np.array_equal(run_attack("Speckle", black, seed=1), black)


@pytest.mark.parametrize(
    ("name", "params"),
    [
        pytest.param("AWGN", {"sigma": 0.001}, id="awgn-lower"),
        pytest.param("AWGN", {"sigma": 0.1}, id="awgn-upper"),
        pytest.param("Impulse", {"density": 0.001}, id="impulse-lower"),
        pytest.param("Impulse", {"density": 0.05}, id="impulse-upper"),
        pytest.param("Periodic", {"amplitude": 0.01, "frequency": 1.0}, id="periodic-lower"),
        pytest.param("Periodic", {"amplitude": 0.2, "frequency": 100.0}, id="periodic-upper"),
        pytest.param("Poisson", {"peak": 1.0}, id="poisson-lower"),
        pytest.param("Poisson", {"peak": 1000.0}, id="poisson-upper"),
        pytest.param("Salt_and_Pepper", {"density": 0.001}, id="salt-pepper-lower"),
        pytest.param("Salt_and_Pepper", {"density": 0.05}, id="salt-pepper-upper"),
        pytest.param("Speckle", {"variance": 0.001}, id="speckle-lower"),
        pytest.param("Speckle", {"variance": 0.05}, id="speckle-upper"),
    ],
)
def test_noise_parameter_boundaries_are_accepted(name, params, small_photo):
    result = attack_class(name).attack(input_image=small_photo, seed=3, **params)
    assert result.shape == small_photo.shape


@pytest.mark.parametrize(
    ("name", "params"),
    [
        pytest.param("AWGN", {"sigma": 0.0}, id="awgn-below"),
        pytest.param("AWGN", {"sigma": 0.101}, id="awgn-above"),
        pytest.param("Impulse", {"density": 0.0}, id="impulse-below"),
        pytest.param("Impulse", {"density": 0.051}, id="impulse-above"),
        pytest.param("Periodic", {"amplitude": 0.0}, id="periodic-amplitude-below"),
        pytest.param("Periodic", {"amplitude": 0.201}, id="periodic-amplitude-above"),
        pytest.param("Periodic", {"frequency": 0.0}, id="periodic-frequency-below"),
        pytest.param("Periodic", {"frequency": 100.1}, id="periodic-frequency-above"),
        pytest.param("Poisson", {"peak": 0.0}, id="poisson-below"),
        pytest.param("Poisson", {"peak": 1000.1}, id="poisson-above"),
        pytest.param("Salt_and_Pepper", {"density": 0.0}, id="salt-pepper-below"),
        pytest.param("Salt_and_Pepper", {"density": 0.051}, id="salt-pepper-above"),
        pytest.param("Speckle", {"variance": 0.0}, id="speckle-below"),
        pytest.param("Speckle", {"variance": 0.051}, id="speckle-above"),
    ],
)
def test_noise_parameters_outside_documented_range_raise(name, params, small_photo):
    with pytest.raises(ValueError):
        attack_class(name).attack(input_image=small_photo, **params)
