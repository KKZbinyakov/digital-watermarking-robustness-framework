"""Registration and category tests for attacks."""

import inspect

import pytest

from dwarf.core.attack_orchestrator.attack_core import (
    Attack_Core,
    Ready_Color_Brightness_Attacks,
    Ready_Compression_Attacks,
    Ready_Filtering_Attacks,
    Ready_Geometric_Attacks,
    Ready_Noise_Attacks,
)

from .helpers import ATTACKS, EXPECTED_ATTACKS

EXPECTED_BY_CATEGORY = {
    Ready_Color_Brightness_Attacks: {
        "Bit_Depth_Reduction",
        "Brightness_Contrast",
        "Color_Jitter",
        "Color_Quantization",
        "Color_Space_Noise",
        "Dithering",
        "Gamma_Correction",
        "Grayscale",
        "Histogram_Equalization",
    },
    Ready_Compression_Attacks: {"Avif", "Bpg", "Flif", "Heic", "Jpeg", "Jpeg2000", "Tiff", "Webp"},
    Ready_Filtering_Attacks: {
        "Anisotropic_Diffusion",
        "Bilateral_Filter",
        "Box_Filter",
        "Gaussian_Blur",
        "Homomorphic_Filter",
        "Median_Filter",
        "Unsharp_Mask",
        "Wiener_Filter",
    },
    Ready_Geometric_Attacks: {"Crop"},
    Ready_Noise_Attacks: {"AWGN", "Impulse", "Periodic", "Poisson", "Salt_and_Pepper", "Speckle"},
}


def test_registry_contains_exactly_the_documented_attacks():
    """A missing Cython build must fail explicitly instead of shrinking parametrization."""
    actual = set(ATTACKS)
    assert actual == EXPECTED_ATTACKS, (
        f"missing attacks: {sorted(EXPECTED_ATTACKS - actual)}; unexpected attacks: {sorted(actual - EXPECTED_ATTACKS)}"
    )


@pytest.mark.parametrize(("category", "expected"), EXPECTED_BY_CATEGORY.items())
def test_attacks_belong_to_the_expected_category(category, expected):
    actual = {name for name, attack in ATTACKS.items() if issubclass(attack, category)}
    assert actual == expected


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_attack_is_reachable_through_the_orchestrator(name):
    attack = Attack_Core.get_attack_class_by_name(name)
    assert attack is ATTACKS[name]
    assert getattr(Attack_Core, name) is attack
    assert not inspect.isabstract(attack)


def test_unknown_attack_name_is_rejected():
    with pytest.raises(KeyError):
        Attack_Core.get_attack_class_by_name("Not_An_Attack")
