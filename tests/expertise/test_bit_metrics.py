"""Exact and boundary tests for BER and NC."""

import pytest

from .helpers import BIT_METRICS, evaluate


@pytest.mark.parametrize(
    "original_bits, extracted_bits, expected",
    [
        ("1010", "1010", 0.0),
        ("1010", "1011", 0.25),
        ("1010", "1001", 0.5),
        ("1010", "0101", 1.0),
        ("11111111", "11111110", 0.125),
        ("1100110011001100", "1100110011001101", 0.0625),
        ("1" * 64, "1" * 63 + "0", 1.0 / 64),
        ("1" * 64, "0" * 32 + "1" * 32, 0.5),
    ],
)
def test_ber_matches_error_fraction(original_bits, extracted_bits, expected):
    assert evaluate("BER", original_bits=original_bits, extracted_bits=extracted_bits) == pytest.approx(expected)


@pytest.mark.parametrize("errors", [0, 1, 7, 16, 31, 32])
def test_ber_counts_every_error(errors):
    length = 32
    original_bits = "0" * length
    extracted_bits = "1" * errors + "0" * (length - errors)

    assert evaluate("BER", original_bits=original_bits, extracted_bits=extracted_bits) == pytest.approx(errors / length)


@pytest.mark.parametrize("length", [8, 16, 64, 256])
def test_full_inversion_reaches_ber_one_and_nc_minus_one(length):
    original_bits = "10" * (length // 2)
    inverted = "".join("1" if bit == "0" else "0" for bit in original_bits)

    assert evaluate("BER", original_bits=original_bits, extracted_bits=inverted) == pytest.approx(1.0)
    assert evaluate("NC", original_bits=original_bits, extracted_bits=inverted) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "original_bits, extracted_bits",
    [
        ("1010", "1010"),
        ("1010", "1011"),
        ("1010", "1001"),
        ("1010", "0101"),
        ("11001010", "10101100"),
    ],
)
def test_nc_and_ber_are_consistent(original_bits, extracted_bits):
    ber = evaluate("BER", original_bits=original_bits, extracted_bits=extracted_bits)
    nc = evaluate("NC", original_bits=original_bits, extracted_bits=extracted_bits)

    assert nc == pytest.approx(1 - 2 * ber)


@pytest.mark.parametrize("name", sorted(BIT_METRICS))
def test_length_mismatch_is_rejected_by_default(name):
    with pytest.raises(ValueError, match="lengths differ"):
        evaluate(name, original_bits="1010", extracted_bits="10101")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("BER", 0.25),
        ("NC", 0.5),
    ],
)
def test_explicit_length_mismatch_compares_common_prefix(name, expected):
    value = evaluate(
        name,
        original_bits="1010",
        extracted_bits="00101",
        allow_length_mismatch=True,
    )

    assert value == pytest.approx(expected)


@pytest.mark.parametrize("name", sorted(BIT_METRICS))
@pytest.mark.parametrize(
    "original_bits, extracted_bits",
    [
        ("", ""),
        ("", "1"),
        ("1", ""),
    ],
)
def test_empty_common_bit_sequence_is_rejected(name, original_bits, extracted_bits):
    with pytest.raises(ValueError):
        evaluate(
            name,
            original_bits=original_bits,
            extracted_bits=extracted_bits,
            allow_length_mismatch=True,
        )


@pytest.mark.parametrize("name", sorted(BIT_METRICS))
@pytest.mark.parametrize("bits", ["10x1", "1 01", "abcd"])
def test_non_binary_characters_are_rejected(name, bits):
    with pytest.raises(ValueError, match="only '0' and '1'"):
        evaluate(name, original_bits=bits, extracted_bits="1011")


@pytest.mark.parametrize("name", sorted(BIT_METRICS))
def test_non_ascii_bit_characters_are_rejected(name):
    with pytest.raises(ValueError):
        evaluate(name, original_bits="10я1", extracted_bits="1011")
