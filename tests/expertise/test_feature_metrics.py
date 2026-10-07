"""Tests for FSIM, FSIMc, and VIF."""

import numpy as np
import pytest

from .helpers import add_noise, evaluate, to_luma


@pytest.mark.parametrize("name", ["FSIM", "FSIMc", "VIF"])
def test_identical_images_have_unit_similarity(name, photo):
    assert evaluate(name, original_image=photo, distorted_image=photo) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("FSIM", 0.9839012255893337),
        ("FSIMc", 0.9741008107684057),
        ("VIF", 0.6369528372056816),
    ],
)
def test_metric_regression_on_fixed_fixture(name, expected, photo, distorted):
    value = evaluate(name, original_image=photo, distorted_image=distorted)

    assert value == pytest.approx(expected, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("name", ["FSIM", "VIF"])
def test_luma_metrics_accept_single_channel_input(name, photo):
    luma = to_luma(photo)

    assert evaluate(name, original_image=luma, distorted_image=luma) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name", ["FSIM", "VIF"])
def test_luma_and_rgb_inputs_agree(name, photo, distorted):
    from_rgb = evaluate(name, original_image=photo, distorted_image=distorted)
    from_luma = evaluate(name, original_image=to_luma(photo), distorted_image=to_luma(distorted))

    assert from_rgb == pytest.approx(from_luma)


def test_fsimc_rejects_single_channel_input(photo):
    with pytest.raises(ValueError):
        evaluate("FSIMc", original_image=to_luma(photo), distorted_image=to_luma(photo))


@pytest.mark.parametrize("name", ["FSIM", "FSIMc", "VIF"])
def test_similarity_decreases_with_noise_strength(name, photo):
    values = [
        evaluate(name, original_image=photo, distorted_image=add_noise(photo, sigma))
        for sigma in (2.0, 8.0, 20.0, 40.0)
    ]

    assert all(values[index] > values[index + 1] for index in range(len(values) - 1)), values


def test_fsimc_detects_chroma_change_that_fsim_ignores(photo):
    luma = 80.0 + to_luma(photo) * (80.0 / 255.0)
    original = np.stack([luma, luma, luma], axis=-1)
    distorted = original.copy()
    distorted[..., 0] += 20.0
    distorted[..., 1] -= 20.0 * 0.299 / 0.587

    fsim = evaluate("FSIM", original_image=original, distorted_image=distorted)
    fsimc = evaluate("FSIMc", original_image=original, distorted_image=distorted)

    assert fsim == pytest.approx(1.0, abs=1e-12)
    assert fsimc < fsim


@pytest.mark.parametrize("name", ["FSIM", "FSIMc", "VIF"])
def test_rejects_malformed_image_matrix(name, photo):
    with pytest.raises(ValueError):
        evaluate(name, original_image=np.zeros((4, 4, 4, 4)), distorted_image=photo)


@pytest.mark.parametrize("name", ["FSIM", "FSIMc", "VIF"])
def test_rejects_different_image_shapes(name, photo):
    with pytest.raises(ValueError):
        evaluate(name, original_image=photo, distorted_image=photo[:-1])


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"block_size": 0}, "block_size must be at least 1"),
        ({"sigma_nsq": 0}, "sigma_nsq must be greater than zero"),
    ],
)
def test_vif_rejects_invalid_parameters(arguments, message, photo):
    with pytest.raises(ValueError, match=message):
        evaluate("VIF", original_image=photo, distorted_image=photo, **arguments)


def test_vif_rejects_image_smaller_than_default_block_minimum(photo):
    tiny = photo[:71, :71]

    with pytest.raises(ValueError, match="at least 72"):
        evaluate("VIF", original_image=tiny, distorted_image=tiny)
