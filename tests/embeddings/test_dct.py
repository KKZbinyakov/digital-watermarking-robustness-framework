"""Recompute DCT article experiments and compare with Xiong 2016 values."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_agreement import assert_max_delta, csv_rows, finite, index_rows, run_evidence


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    result_dir, paper = run_evidence("DCT", tmp_path_factory)
    channels = index_rows(csv_rows(result_dir, "table1.csv"), ("image", "channel"))
    attacks = index_rows(csv_rows(result_dir, "summary.csv"), ("image", "attack"))
    expected_channels = {(image, channel) for image in paper["table1"] for channel in "RGB"}
    expected_attacks = {(image, attack) for image, values in paper["table2"].items() for attack in values}
    assert set(channels) == expected_channels
    assert expected_attacks <= set(attacks)
    return paper, channels, attacks


def test_table1_channel_psnr(evidence):
    """All 12 RGB PSNR values."""
    paper, channels, _ = evidence
    assert_max_delta(
        (
            (f"{image}/{channel}", channels[image, channel]["measured_psnr_db"], reference)
            for image, values in paper["table1"].items()
            for channel, reference in values.items()
        ),
        1.0,
        "DCT Table 1 PSNR (dB)",
    )


@pytest.mark.parametrize("attack_group,limit", [("other", 1.78), ("gaussian", 18.4)])
def test_table2_attack_psnr(evidence, attack_group, limit):
    """All Table 2 attack PSNR points."""
    paper, _, attacks = evidence
    assert_max_delta(
        (
            (f"{image}/{attack}", attacks[image, attack]["psnr_db_mean"], values["psnr_db"])
            for image, references in paper["table2"].items()
            for attack, values in references.items()
            if (attack == "gaussian") == (attack_group == "gaussian")
        ),
        limit,
        f"DCT Table 2 {attack_group} PSNR (dB)",
    )


@pytest.mark.parametrize("attack_group,limit", [("other", 0.065), ("gaussian", 0.214)])
def test_table2_attack_nc(evidence, attack_group, limit):
    """Compare published NC to plotted bit accuracy, including all four JPEG points per host."""
    paper, _, attacks = evidence
    assert_max_delta(
        (
            (f"{image}/{attack}", attacks[image, attack]["accuracy_mean"], values["nc"])
            for image, references in paper["table2"].items()
            for attack, values in references.items()
            if (attack == "gaussian") == (attack_group == "gaussian")
        ),
        limit,
        f"DCT Table 2 {attack_group} NC/accuracy",
    )


def test_reported_bit_accuracy_matches_ber(evidence):
    """The measured NC proxy must represent 1-BER for every attack point."""
    _, _, attacks = evidence
    for key, row in attacks.items():
        if row["attack"] == "clean":
            continue
        assert abs(finite(row["accuracy_mean"]) + finite(row["ber_mean"]) - 1) < 1e-12, key
