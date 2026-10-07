"""Targeted contract and robustness tests for the 2025 ADC-ABW method."""

from pathlib import Path

import numpy as np
import pytest
from conftest import make_photo
from PIL import Image

import dwarf.ready_solutions.embedding_solutions  # noqa: F401
from dwarf import Embedding_Core
from dwarf.core.embedding_orchestrator.embedding_core import Ready_Spread_Spectrum_Embeddings
from dwarf.ready_solutions.attack_solutions.color_brightness.brightness_contrast import Brightness_Contrast
from dwarf.ready_solutions.attack_solutions.compression.jpeg import Jpeg
from dwarf.ready_solutions.attack_solutions.filtering.gaussian_blur import Gaussian_Blur
from dwarf.ready_solutions.attack_solutions.noise.awgn import AWGN
from dwarf.ready_solutions.embedding_solutions.spread_spectrum.adc_abw import ADC_ABW


def embed(image, bits, **parameters):
    """Embeds the test payload using ADC-ABW defaults."""
    return ADC_ABW.embedding(input_image=image, watermark_bits=bits, **parameters)


def extract(image, num_bits, **parameters):
    """Extracts the test payload using ADC-ABW defaults."""
    return ADC_ABW.extraction(input_image=image, num_bits=num_bits, **parameters)


@pytest.mark.parametrize(
    "bits",
    [
        np.zeros(16, dtype=np.uint8),
        np.ones(16, dtype=np.uint8),
        np.resize(np.array([0, 1], dtype=np.uint8), 16),
        np.random.default_rng(7).integers(0, 2, 16, dtype=np.uint8),
    ],
    ids=["zeros", "ones", "alternating", "random"],
)
def test_clean_round_trip(bits):
    image = make_photo(128, 128, seed=11)

    watermarked = embed(image, bits, key=2026)
    extracted = extract(watermarked, bits.size, key=2026)

    assert np.array_equal(extracted.astype(np.uint8), bits)


def test_contract_registration_determinism_and_input_immutability():
    image = np.ascontiguousarray(make_photo(128, 128, seed=13))
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    image_before = image.copy()
    bits_before = bits.copy()

    first = embed(image, bits, key=19)
    second = embed(image, bits, key=19)
    extracted = extract(first, bits.size, key=19)

    assert Embedding_Core.get_embedding_class_by_name("ADC_ABW") is ADC_ABW
    assert issubclass(ADC_ABW, Ready_Spread_Spectrum_Embeddings)
    assert first.dtype == np.uint8
    assert first.shape == image.shape
    assert first.flags.c_contiguous
    assert extracted.dtype == np.int8
    assert extracted.shape == bits.shape
    assert extracted.flags.c_contiguous
    assert np.array_equal(first, second)
    assert np.array_equal(image, image_before)
    assert np.array_equal(bits, bits_before)


def test_code_capacity_and_image_capacity_are_checked_separately():
    image = make_photo(64, 64, seed=17)
    payload = np.resize(np.array([0, 1], dtype=np.uint8), 32)

    watermarked = embed(image, payload, spreading_length=128, num_blocks=32)

    assert np.array_equal(
        extract(watermarked, payload.size, spreading_length=128, num_blocks=32).astype(np.uint8),
        payload,
    )
    with pytest.raises(ValueError, match="payload capacity"):
        embed(image, np.resize(payload, 65), spreading_length=128, num_blocks=32)
    with pytest.raises(ValueError, match="spreading_length is too large"):
        embed(make_photo(30, 34, seed=18), payload, spreading_length=128, num_blocks=8)
    with pytest.raises(ValueError, match="payload capacity"):
        extract(watermarked, 65, spreading_length=128, num_blocks=32)
    with pytest.raises(ValueError, match="spreading_length is too large"):
        extract(image, payload.size, spreading_length=2048, num_blocks=32)
    with pytest.raises(ValueError, match="num_blocks is too large"):
        embed(make_photo(127, 256, seed=19), payload)


def test_rejects_zero_energy_embedding_location():
    image = np.full((128, 128, 3), 127, dtype=np.uint8)
    bits = np.array([1, 0, 1, 0], dtype=np.uint8)

    with pytest.raises(ValueError, match="no energy"):
        embed(image, bits)


def test_rejects_no_op_when_host_already_decodes_as_payload():
    image = make_photo(256, 256, seed=18)
    bits = extract(image, 64, key=2026).astype(np.uint8)

    with pytest.raises(ValueError, match="quantization"):
        embed(image, bits, key=2026, attack_angle=89.999)


def test_default_profile_survives_uint8_quantization_on_natural_image():
    image_path = Path(__file__).parents[2] / "Asuka.jpg"
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    bits = np.random.default_rng(19).integers(0, 2, 64, dtype=np.uint8)

    watermarked = embed(image, bits, key=2026)

    assert not np.array_equal(watermarked, image)
    assert np.array_equal(extract(watermarked, bits.size, key=2026).astype(np.uint8), bits)


@pytest.mark.parametrize(
    ("image", "error"),
    [
        (np.zeros((64, 64), dtype=np.uint8), ValueError),
        (np.zeros((64, 64, 4), dtype=np.uint8), ValueError),
        (np.zeros((64, 64, 3), dtype=np.float32), TypeError),
    ],
    ids=["grayscale", "rgba", "float"],
)
def test_rejects_invalid_image_contract(image, error):
    bits = np.array([1], dtype=np.uint8)

    with pytest.raises(error):
        embed(image, bits)
    with pytest.raises(error):
        extract(image, 1)


@pytest.mark.parametrize(
    ("bits", "error"),
    [
        (np.empty(0, dtype=np.uint8), ValueError),
        (np.zeros((1, 2), dtype=np.uint8), ValueError),
        (np.array([0, 1], dtype=np.int64), TypeError),
        (np.array([0, 2], dtype=np.uint8), ValueError),
    ],
    ids=["empty", "two-dimensional", "wrong-dtype", "non-binary"],
)
def test_rejects_invalid_payload_contract(bits, error):
    with pytest.raises(error):
        embed(make_photo(64, 64, seed=21), bits)


@pytest.mark.parametrize(
    ("parameter", "value", "error"),
    [
        ("attack_angle", 0.0, ValueError),
        ("attack_angle", 90.0, ValueError),
        ("attack_angle", np.inf, ValueError),
        ("spreading_length", 1, ValueError),
        ("spreading_length", 256.0, TypeError),
        ("num_blocks", 0, ValueError),
        ("num_blocks", 32.0, TypeError),
        ("key", -1, ValueError),
    ],
)
def test_rejects_invalid_embedding_parameters(parameter, value, error):
    image = make_photo(64, 64, seed=23)
    bits = np.array([1], dtype=np.uint8)

    with pytest.raises(error):
        embed(image, bits, **{parameter: value})


@pytest.mark.parametrize(
    ("parameter", "value", "error"),
    [
        ("num_bits", 0, ValueError),
        ("num_bits", -1, ValueError),
        ("num_bits", 1.0, TypeError),
        ("spreading_length", 1, ValueError),
        ("num_blocks", 0, ValueError),
        ("key", -1, ValueError),
    ],
)
def test_rejects_invalid_extraction_parameters(parameter, value, error):
    arguments = {"num_bits": 1, parameter: value}

    with pytest.raises(error):
        ADC_ABW.extraction(input_image=make_photo(64, 64, seed=27), **arguments)


def test_can_extract_payload_prefix():
    image = make_photo(128, 128, seed=28)
    bits = np.random.default_rng(29).integers(0, 2, 24, dtype=np.uint8)

    watermarked = embed(image, bits, key=101)

    assert np.array_equal(extract(watermarked, 12, key=101).astype(np.uint8), bits[:12])


def test_non_default_block_grid_handles_non_divisible_dimensions():
    image = make_photo(131, 139, seed=30)
    bits = np.random.default_rng(31).integers(0, 2, 24, dtype=np.uint8)

    watermarked = embed(image, bits, key=102, num_blocks=64)

    assert watermarked.shape == image.shape
    assert np.array_equal(extract(watermarked, bits.size, key=102, num_blocks=64).astype(np.uint8), bits)


def test_smaller_attack_angle_increases_embedding_distortion():
    image = make_photo(256, 256, seed=32)
    bits = np.random.default_rng(33).integers(0, 2, 32, dtype=np.uint8)

    robust = embed(image, bits, key=103, attack_angle=12.0)
    subtle = embed(image, bits, key=103, attack_angle=45.0)
    robust_mse = np.mean((image.astype(np.float64) - robust.astype(np.float64)) ** 2)
    subtle_mse = np.mean((image.astype(np.float64) - subtle.astype(np.float64)) ** 2)

    assert robust_mse > subtle_mse
    assert np.array_equal(extract(robust, bits.size, key=103).astype(np.uint8), bits)
    assert np.array_equal(extract(subtle, bits.size, key=103).astype(np.uint8), bits)


def test_wrong_key_does_not_recover_payload():
    image = make_photo(256, 256, seed=29)
    bits = np.random.default_rng(31).integers(0, 2, 64, dtype=np.uint8)

    watermarked = embed(image, bits, key=1234)
    correct = extract(watermarked, bits.size, key=1234)
    wrong = extract(watermarked, bits.size, key=4321)

    assert np.array_equal(correct.astype(np.uint8), bits)
    assert np.mean(wrong.astype(np.uint8) != bits) >= 0.25


def test_visual_quality():
    image = make_photo(256, 256, seed=37)
    bits = np.random.default_rng(41).integers(0, 2, 64, dtype=np.uint8)

    watermarked = embed(image, bits, key=73)

    mse = np.mean((image.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
    psnr = np.inf if mse == 0.0 else 10.0 * np.log10(255.0**2 / mse)

    assert psnr >= 40.0


def test_published_512_profile_regression_under_direct_scaling():
    image = make_photo(512, 512, seed=42)
    bits = np.random.default_rng(43).integers(0, 2, 64, dtype=np.uint8)
    watermarked = embed(image, bits, key=78)

    for factor in (0.7, 0.9, 1.4, 1.8):
        side = round(image.shape[0] * factor)
        attacked = np.asarray(Image.fromarray(watermarked).resize((side, side), Image.Resampling.BICUBIC))
        extracted = extract(attacked, bits.size, key=78)

        assert np.mean(extracted.astype(np.uint8) != bits) <= 0.05


@pytest.mark.parametrize(
    ("attack", "parameters"),
    [
        pytest.param(Jpeg, {"quality": 75}, id="jpeg"),
        pytest.param(AWGN, {"sigma": 0.01, "seed": 5}, id="awgn"),
        pytest.param(Gaussian_Blur, {"sigma": 1.0}, id="gaussian-blur"),
        pytest.param(Brightness_Contrast, {"brightness": 1.2, "contrast": 1.2}, id="brightness-contrast"),
    ],
)
def test_default_profile_regression_under_mild_signal_attacks(attack, parameters):
    image = make_photo(256, 256, seed=43)
    bits = np.random.default_rng(47).integers(0, 2, 64, dtype=np.uint8)
    watermarked = embed(image, bits, key=79)

    attacked = attack.attack(input_image=watermarked, **parameters)
    extracted = extract(attacked, bits.size, key=79)
    ber = np.mean(extracted.astype(np.uint8) != bits)

    assert ber <= 0.10
