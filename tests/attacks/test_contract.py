"""Shared behavioural contract for every bundled attack."""

import numpy as np
import pytest

from .helpers import (
    EXPECTED_ATTACKS,
    LOSSLESS_ATTACKS,
    SEEDED_ATTACKS,
    attack_cases,
    run_attack,
)

IN_PROCESS_ATTACKS = EXPECTED_ATTACKS - {
    "Avif",
    "Bpg",
    "Flif",
    "Heic",
    "Jpeg",
    "Jpeg2000",
    "Tiff",
    "Webp",
}


@pytest.mark.parametrize("name", attack_cases())
def test_default_attack_returns_writable_rgb_without_mutating_input(name, small_photo):
    before = small_photo.copy()

    result = run_attack(name, small_photo)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == small_photo.shape
    assert result.flags.writeable
    assert np.array_equal(small_photo, before), f"{name} mutated its input in place"


@pytest.mark.parametrize("name", attack_cases())
@pytest.mark.parametrize(
    "convert",
    [
        pytest.param(lambda image: image.astype(np.float64) + 0.25, id="float64"),
        pytest.param(lambda image: image.tolist(), id="list"),
    ],
)
def test_attack_accepts_supported_array_like_inputs(name, convert, small_photo):
    result = run_attack(name, convert(small_photo))
    assert result.shape == small_photo.shape
    assert result.dtype == np.uint8


@pytest.mark.parametrize("name", attack_cases())
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(np.zeros((8, 8), dtype=np.uint8), id="grayscale"),
        pytest.param(np.zeros((8, 8, 4), dtype=np.uint8), id="rgba"),
    ],
)
def test_attack_rejects_non_rgb_matrices(name, invalid):
    with pytest.raises(ValueError, match="RGB|shape|image"):
        run_attack(name, invalid)


@pytest.mark.parametrize("name", attack_cases(IN_PROCESS_ATTACKS))
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (3, 5)])
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning:scipy.signal._signaltools")
def test_in_process_attack_handles_tiny_rectangular_frames(name, shape):
    height, width = shape
    image = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)

    result = run_attack(name, image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


@pytest.mark.parametrize("name", attack_cases(EXPECTED_ATTACKS - SEEDED_ATTACKS))
def test_deterministic_attack_repeats_bit_exactly(name, small_photo):
    assert np.array_equal(run_attack(name, small_photo), run_attack(name, small_photo))


@pytest.mark.parametrize("name", attack_cases(SEEDED_ATTACKS))
def test_seed_makes_random_attack_reproducible(name, small_photo):
    assert np.array_equal(
        run_attack(name, small_photo, seed=11),
        run_attack(name, small_photo, seed=11),
    )


@pytest.mark.parametrize("name", attack_cases(SEEDED_ATTACKS))
def test_different_seeds_change_random_attack(name, small_photo):
    assert not np.array_equal(
        run_attack(name, small_photo, seed=1),
        run_attack(name, small_photo, seed=2),
    )


@pytest.mark.parametrize("name", attack_cases(LOSSLESS_ATTACKS))
def test_lossless_attack_is_bit_exact(name, photo):
    assert np.array_equal(run_attack(name, photo), photo)


@pytest.mark.parametrize("name", attack_cases(EXPECTED_ATTACKS - LOSSLESS_ATTACKS))
def test_default_attack_has_an_observable_effect(name, photo):
    assert not np.array_equal(run_attack(name, photo), photo)


@pytest.mark.parametrize("name", attack_cases())
def test_attack_output_can_feed_another_attack(name, small_photo):
    attacked = run_attack(name, small_photo)
    recompressed = run_attack("Jpeg", attacked)

    assert recompressed.shape == attacked.shape
    assert recompressed.dtype == np.uint8
