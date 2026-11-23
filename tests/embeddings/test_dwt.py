"""Recompute DWT article experiments and compare every NCQM value."""

from pathlib import Path

import pytest

from ._paper_support import assert_max_delta, csv_rows, index_rows, run_evidence

SPEC = Path(__file__).resolve().parents[2] / "examples" / "paper_verification" / "dwt.yaml"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    result_dir, spec = run_evidence(SPEC, tmp_path_factory)
    paper, agreement = spec["reference"], spec["agreement"]
    quality = index_rows(csv_rows(result_dir, "quality.csv"), ("image",))
    attacks = index_rows(csv_rows(result_dir, "summary.csv"), ("image", "attack"))
    assert {image for (image,) in quality} == set(paper["mssim_table7_text"])
    expected = {
        (image, attack)
        for table in ("table7", "table8")
        for image, values in paper[table].items()
        for attack in values["NCQM"]
    }
    assert expected <= set(attacks)
    return paper, agreement, quality, attacks


def test_quality_psnr_42_db(evidence):
    """Four article hosts are calibrated to 42 dB."""
    _, agreement, quality, _ = evidence
    target = agreement["clean_psnr_db"]
    assert_max_delta(
        ((image, row["psnr_db"], target["target"]) for (image,), row in quality.items()),
        target["limit"],
        "DWT clean PSNR (dB)",
    )


def test_quality_mssim(evidence):
    """All four published MSSIM values."""
    paper, agreement, quality, _ = evidence
    assert_max_delta(
        ((image, quality[(image,)]["mssim"], reference) for image, reference in paper["mssim_table7_text"].items()),
        agreement["mssim"]["limit"],
        "DWT MSSIM",
    )


@pytest.mark.parametrize("table", ("table7", "table8"))
def test_ber_tables(evidence, table):
    """Compare every NCQM BER with the run in percentage points, not BER fractions."""
    paper, agreement, _, attacks = evidence
    assert_max_delta(
        (
            (f"{image}/{attack}", 100 * float(attacks[image, attack]["ber_mean"]), reference)
            for image, values in paper[table].items()
            for attack, reference in values["NCQM"].items()
        ),
        agreement["ber_percentage_points"][table],
        f"DWT {table} BER (percentage points)",
    )
