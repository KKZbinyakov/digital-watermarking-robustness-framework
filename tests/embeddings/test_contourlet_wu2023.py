"""Equation and numerical-reproduction tests for Wu et al. (2023)."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dwarf.ready_solutions.embedding_solutions.frequency.contourlet_wu2023 import (
    Contourlet_Wu2023,
    _adaptive_steps,
    _arnold,
    _embed_lowpass,
    _extract_lowpass,
    _full_bounds_transform,
    _lowpass_analysis,
    _lowpass_synthesis_delta,
    _moment_transform,
    _normalized_luma,
)

AUTHOR_HOSTS = {
    "Lena": ("4.2.04.tiff", "c056da23302d2fb0d946e7ffa11e0d94618224193ff6e2f78ef8097bb8a3569b"),
    "Mandrill": ("4.2.03.tiff", "3f590b52279fb59b81906f1e928ae713a5357b1afc1a2017a103adb563fb4494"),
    "Airplane": ("4.2.05.tiff", "ae9eba12b80df955e47585bf4a16332ea0dadc753668d3457c509564934dbd40"),
    "Fishingboat": ("boat.512.tiff", "6bc7319b1e3b44ce0b2a571ac39a28bac94437d812057d1afb2cf78ea3a9a35f"),
    "House": ("house.tiff", "82717baf56d3f48bc4e2767afc9f233e7f595ad150ca91932a4a8daa46b3300f"),
    "Bridge": ("5.2.10.tiff", "044ed17593bc94a5c0fff638f2c4d068f563d415980e1f61b423c778bb61c181"),
}
PAPER_ROTATION_NC = {10: 0.9281, 30: 0.8252, 50: 0.8993, 70: 0.8760, 90: 0.8605}


def _binary_moment(weights, x, y, p, q):
    return np.sum(weights * x**p * y**q) / np.sum(weights)


def test_adaptive_step_uses_published_energy_formula():
    block = np.arange(1.0, 17.0).reshape(1, 4, 4)
    energy = np.mean(block**2)

    actual = _adaptive_steps(block, 48.0)[0]

    assert actual == pytest.approx(np.log2((energy + 1000.0) / 1000.0) * 48.0)


def test_pkva_lowpass_delta_is_recovered_by_analysis():
    delta = np.random.default_rng(7).normal(size=(32, 32))

    recovered = _lowpass_analysis(_lowpass_synthesis_delta(delta))

    np.testing.assert_allclose(recovered, delta, rtol=1e-12, atol=1e-12)


def test_content_adaptive_qim_is_self_consistent():
    lowpass = np.random.default_rng(17).uniform(20.0, 400.0, size=(128, 128))
    bits = np.random.default_rng(3).integers(0, 2, 32 * 32, dtype=np.uint8)

    marked = _embed_lowpass(lowpass, bits, 48.0, 4)
    extracted = _extract_lowpass(marked, bits.size, 48.0, 4)

    np.testing.assert_array_equal(extracted, bits)


def test_arnold_scrambling_round_trip_with_paper_matrix_parameters():
    bits = np.random.default_rng(3).integers(0, 2, 32 * 32, dtype=np.uint8)

    scrambled = _arnold(bits, 11, matrix=(1, 2, 3, 7))
    restored = _arnold(scrambled, 11, inverse=True, matrix=(1, 2, 3, 7))

    assert not np.array_equal(scrambled, bits)
    np.testing.assert_array_equal(restored, bits)


def test_four_stage_normalization_satisfies_published_moment_constraints():
    yy, xx = np.indices((73, 81), dtype=np.float64)
    weights = np.exp(-((xx - 23.0) ** 2 / 500.0 + (yy - 41.0) ** 2 / 260.0))
    weights += 0.4 * np.exp(-((xx - 65.0) ** 2 / 80.0 + (yy - 14.0) ** 2 / 130.0))
    target_std = 37.0
    output_shape = (97, 101)

    transform = _moment_transform(weights, target_std, output_shape)
    transformed_x = transform[0, 0] * xx + transform[0, 1] * yy + transform[0, 2]
    transformed_y = transform[1, 0] * xx + transform[1, 1] * yy + transform[1, 2]
    transformed_x -= (output_shape[1] - 1.0) / 2.0
    transformed_y -= (output_shape[0] - 1.0) / 2.0

    assert _binary_moment(weights, transformed_x, transformed_y, 2, 0) == pytest.approx(target_std**2)
    assert _binary_moment(weights, transformed_x, transformed_y, 0, 2) == pytest.approx(target_std**2)
    assert _binary_moment(weights, transformed_x, transformed_y, 1, 1) == pytest.approx(0.0, abs=1e-9)
    radial_third = _binary_moment(weights, transformed_x, transformed_y, 3, 0) + _binary_moment(
        weights, transformed_x, transformed_y, 1, 2
    )
    assert radial_third == pytest.approx(0.0, abs=1e-7)


def test_full_bounds_transform_contains_every_transformed_corner():
    transform = np.array(
        [[1.3, -0.4, 17.2], [0.25, 0.9, -8.7], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    adjusted, output_shape = _full_bounds_transform((73, 81), transform)
    corners = np.array([[0, 0, 1], [80, 0, 1], [0, 72, 1], [80, 72, 1]], dtype=np.float64)

    mapped = (adjusted @ corners.T).T[:, :2]

    assert np.all(mapped >= 0.0)
    assert np.all(mapped[:, 0] <= output_shape[1] - 1.0)
    assert np.all(mapped[:, 1] <= output_shape[0] - 1.0)


def test_positive_sign_full_bounds_normalization_is_rotation_invariant():
    yy, xx = np.indices((96, 96), dtype=np.float64)
    gray = 40.0 + 160.0 * np.exp(-((xx - 25.0) ** 2 / 300.0 + (yy - 58.0) ** 2 / 180.0))
    gray += 55.0 * np.exp(-((xx - 73.0) ** 2 / 90.0 + (yy - 19.0) ** 2 / 120.0))
    pixels = np.clip(np.floor(gray + 0.5), 0.0, 255.0).astype(np.uint8)
    image = np.repeat(pixels[..., None], 3, axis=2)

    normalized, _ = _normalized_luma(
        image,
        True,
        48.0,
        (128, 128),
        bounds_mode="full",
        sign_mode="positive",
    )
    rotated, _ = _normalized_luma(
        np.ascontiguousarray(np.rot90(image)),
        True,
        48.0,
        (128, 128),
        bounds_mode="full",
        sign_mode="positive",
    )

    assert rotated.shape == normalized.shape
    np.testing.assert_allclose(rotated, normalized, rtol=0.0, atol=1e-9)


def test_paper_average_psnr_calibration_on_author_hosts():
    """Run the external author-image audit when its assets are supplied."""
    assets_value = os.environ.get("DWARF_WU2023_AUTHOR_ASSETS")
    if not assets_value:
        pytest.skip("set DWARF_WU2023_AUTHOR_ASSETS to run the author-image reproduction")
    assets = Path(assets_value)
    bits = np.random.default_rng(42).integers(0, 2, 32 * 32, dtype=np.uint8)
    values = []
    marked_lena = None
    params = {
        "coefficient_scale": 4.710,
        "centroid_radius": 512.0,
        "normalization_bounds": "full",
        "normalization_sign_mode": "positive",
    }

    for name, (filename, expected_hash) in AUTHOR_HOSTS.items():
        path = assets / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        image = np.repeat(gray[..., None], 3, axis=2)
        marked = Contourlet_Wu2023.embedding(
            input_image=image,
            watermark_bits=bits,
            compensation_iterations=3,
            **params,
        )
        mse = np.mean((image[..., 0].astype(np.float64) - marked[..., 0].astype(np.float64)) ** 2)
        values.append(10.0 * np.log10(255.0**2 / mse))
        if name == "Lena":
            marked_lena = marked

    assert np.mean(values) == pytest.approx(40.5020, abs=0.01)
    assert marked_lena is not None

    actual_rotation_nc = []
    for angle in PAPER_ROTATION_NC:
        gray = Image.fromarray(marked_lena[..., 0], mode="L")
        attacked_gray = np.asarray(
            gray.rotate(angle, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0),
            dtype=np.uint8,
        )
        attacked = np.repeat(attacked_gray[..., None], 3, axis=2)
        extracted = Contourlet_Wu2023.extraction(
            input_image=np.ascontiguousarray(attacked),
            num_bits=bits.size,
            **params,
        )
        source = bits.astype(np.float64)
        recovered = extracted.astype(np.float64)
        denominator = np.sqrt(np.sum(source * source) * np.sum(recovered * recovered))
        actual_rotation_nc.append(float(np.dot(source, recovered) / denominator))

    expected_rotation_nc = np.array(list(PAPER_ROTATION_NC.values()))
    assert np.mean(np.abs(np.array(actual_rotation_nc) - expected_rotation_nc)) < 0.08
    assert actual_rotation_nc[3] >= PAPER_ROTATION_NC[70]
