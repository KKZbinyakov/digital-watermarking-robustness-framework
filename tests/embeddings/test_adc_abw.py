"""Regression tests for Guo et al.'s 2025 ADC-ABW4 profile."""

import io
from pathlib import Path

import numpy as np
import pytest
from conftest import make_photo
from PIL import Image
from scipy.fft import dctn

import dwarf.ready_solutions.embedding_solutions  # noqa: F401
from dwarf import Embedding_Core
from dwarf.ready_solutions.embedding_solutions.spread_spectrum.adc_abw import (
    ADC_ABW,
    _attack_based_weight,
    _code_matrix,
    _dc_matrix,
    _dc_spectrum,
    _location,
    _to_luma,
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


def test_adc_step_1_matches_an_explicit_orthonormal_block_dct():
    """Section 4.2, step 1: D1 consists of the blocks' DCT DC terms."""
    channel = np.random.default_rng(11).normal(size=(12, 18))

    dc, row_edges, column_edges, _ = _dc_matrix(channel, 3)
    expected = np.empty((3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            block = channel[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ]
            expected[row, column] = dctn(block, norm="ortho")[0, 0]

    np.testing.assert_allclose(dc, expected, rtol=1e-12, atol=1e-12)


def test_adc_step_2_applies_a_second_orthonormal_dct():
    """Section 4.2, step 2: D2 is the DCT of the D1 matrix."""
    image = make_photo(32, 48, seed=12)
    dc, _, _, _ = _dc_matrix(_to_luma(image), 8)

    np.testing.assert_allclose(
        _dc_spectrum(image, 8),
        dctn(dc, norm="ortho"),
        rtol=1e-12,
        atol=1e-12,
    )


def test_adc_steps_3_and_4_select_the_end_of_the_zigzag_sequence():
    """Section 4.2, steps 3-4: the location is the high-frequency tail."""
    spectrum = np.arange(16, dtype=np.float64).reshape(4, 4)

    location, indices = _location(spectrum, spreading_length=5)

    expected_indices = np.array([10, 7, 11, 14, 15])
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(location, spectrum.reshape(-1)[expected_indices])


def test_many_to_one_codes_satisfy_equations_4_and_6():
    """Orthogonal Gaussian codes spread and recover every bipolar bit."""
    spreading_length = 320
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    bipolar = 2.0 * bits.astype(np.float64) - 1.0
    codes = _code_matrix(spreading_length, key=3743139, num_codes=bits.size)

    np.testing.assert_allclose(
        codes @ codes.T,
        spreading_length * np.eye(bits.size),
        rtol=1e-12,
        atol=1e-12,
    )
    spread_watermark = bipolar @ codes
    np.testing.assert_allclose(
        np.linalg.norm(spread_watermark),
        np.sqrt(spreading_length * bits.size),
        rtol=1e-12,
    )
    np.testing.assert_allclose(codes @ spread_watermark / spreading_length, bipolar, atol=1e-12)

    # A payload prefix must use exactly the same secret modulation signals.
    prefix_codes = _code_matrix(spreading_length, key=3743139, num_codes=4)
    np.testing.assert_allclose(prefix_codes, codes[:4], rtol=0.0, atol=1e-14)
    assert not np.array_equal(codes, _code_matrix(spreading_length, key=3743140, num_codes=bits.size))


def test_abw_equation_18_realizes_the_requested_attack_angle():
    """For the paper's orthogonality assumption, Equation (18) gives angle A."""
    location = np.array([3.0, 4.0, 0.0])
    spread_watermark = np.array([0.0, 0.0, 2.0])
    attack_angle = 30.0

    weight = _attack_based_weight(location, spread_watermark, attack_angle)
    watermarked_location = location + weight * spread_watermark
    realized_angle = np.rad2deg(
        np.arccos(
            np.dot(watermarked_location, spread_watermark)
            / (np.linalg.norm(watermarked_location) * np.linalg.norm(spread_watermark))
        )
    )

    np.testing.assert_allclose(realized_angle, attack_angle, rtol=0.0, atol=1e-12)
    assert _attack_based_weight(location, spread_watermark, 12.0) > weight


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


def test_defaults_are_the_published_adc_abw4_profile():
    image = make_photo(256, 256, seed=14)
    bits = np.random.default_rng(20).integers(0, 2, 64, dtype=np.uint8)

    default = _embed(image, bits)
    explicit = _embed(
        image,
        bits,
        attack_angle=30.0,
        spreading_length=320,
        num_blocks=128,
        key=0,
    )

    np.testing.assert_array_equal(default, explicit)
    np.testing.assert_array_equal(_extract(default, 64), bits)


def test_y_channel_embedding_preserves_rgb_chroma_differences():
    """Adding one luma delta to R, G, and B is the linear Y-only operation."""
    image = np.random.default_rng(21).integers(64, 192, size=(256, 256, 3), dtype=np.uint8)
    bits = np.random.default_rng(22).integers(0, 2, 32, dtype=np.uint8)

    watermarked = _embed(image, bits, key=23, spreading_length=128, num_blocks=64)
    delta = watermarked.astype(np.int16) - image.astype(np.int16)

    np.testing.assert_array_equal(delta[..., 0], delta[..., 1])
    np.testing.assert_array_equal(delta[..., 1], delta[..., 2])


@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        ({"attack_angle": 0.0}, ValueError, "between 0 and 90"),
        ({"attack_angle": 90.0}, ValueError, "between 0 and 90"),
        ({"attack_angle": np.nan}, ValueError, "finite"),
        ({"attack_angle": True}, TypeError, "real number"),
        ({"spreading_length": 1}, ValueError, "spreading_length must be >= 2"),
        ({"num_blocks": 0}, ValueError, "num_blocks must be >= 1"),
        ({"key": -1}, ValueError, "key must be >= 0"),
    ],
)
def test_embedding_rejects_invalid_profile_parameters(parameters, error, message):
    image = make_photo(32, 32, seed=24)
    bits = np.array([0, 1], dtype=np.uint8)

    with pytest.raises(error, match=message):
        _embed(image, bits, **parameters)


def test_public_contract_rejects_invalid_images_payloads_and_capacities():
    image = make_photo(32, 32, seed=25)

    with pytest.raises(ValueError, match="shape"):
        _embed(image[..., 0], np.array([0], dtype=np.uint8), num_blocks=8)
    with pytest.raises(TypeError, match="dtype uint8"):
        _embed(image.astype(np.float64), np.array([0], dtype=np.uint8), num_blocks=8)
    with pytest.raises(TypeError, match="dtype uint8"):
        _embed(image, np.array([0, 1]), num_blocks=8)
    with pytest.raises(ValueError, match="must not be empty"):
        _embed(image, np.array([], dtype=np.uint8), num_blocks=8)
    with pytest.raises(ValueError, match="only 0 and 1"):
        _embed(image, np.array([0, 2], dtype=np.uint8), num_blocks=8)
    with pytest.raises(ValueError, match="Not enough payload capacity"):
        _embed(image, np.zeros(65, dtype=np.uint8), spreading_length=320, num_blocks=8)
    with pytest.raises(ValueError, match="num_blocks is too large"):
        _embed(image, np.array([0], dtype=np.uint8), num_blocks=33)
    with pytest.raises(ValueError, match="spreading_length is too large"):
        _embed(image, np.array([0], dtype=np.uint8), spreading_length=65, num_blocks=8)
    with pytest.raises(ValueError, match="selected ADC location has no energy"):
        _embed(
            np.full((32, 32, 3), 128, dtype=np.uint8),
            np.array([0], dtype=np.uint8),
            spreading_length=32,
            num_blocks=8,
        )
    with pytest.raises(ValueError, match="payload capacity is 64"):
        _extract(image, 65, spreading_length=320, num_blocks=8)


def test_adc_abw4_surrogate_tracks_the_published_quality_scale():
    """A 512x512 photograph gives the paper's approximately 48 dB scale.

    This is a deterministic surrogate, not a replacement for the paper's
    undisclosed selection of 100 UW/SIPI files, watermark, PRNG, or MATLAB codec.
    """
    image_path = Path(__file__).parents[2] / "Asuka.jpg"
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS))
    bits = np.random.default_rng(3743139).integers(0, 2, 64, dtype=np.uint8)

    watermarked = _embed(image, bits)
    clean = _extract(watermarked, bits.size)
    jpeg_90 = _extract(_jpeg(watermarked, 90), bits.size)
    jpeg_50 = _extract(_jpeg(watermarked, 50), bits.size)

    # Table 2 reports a 48.1 dB average for 100 images. This one-image result is
    # only a scale/regression check, but uses the paper's 4x4 block geometry.
    assert abs(_psnr(image, watermarked) - 48.1) <= 1.0
    assert _ber(bits, clean) == 0.0
    assert _ber(bits, jpeg_90) <= 1 / 64
    assert _ber(bits, jpeg_50) <= 2 / 64


def test_smaller_attack_angle_increases_distortion():
    image = make_photo(256, 256, seed=32)
    bits = np.random.default_rng(33).integers(0, 2, 32, dtype=np.uint8)

    robust = _embed(image, bits, key=103, attack_angle=12.0)
    subtle = _embed(image, bits, key=103, attack_angle=45.0)

    assert _psnr(image, robust) < _psnr(image, subtle)
    np.testing.assert_array_equal(_extract(robust, bits.size, key=103), bits)
    np.testing.assert_array_equal(_extract(subtle, bits.size, key=103), bits)
