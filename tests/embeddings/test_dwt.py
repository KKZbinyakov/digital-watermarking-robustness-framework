"""Recompute DWT article experiments and compare every NCQM value."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_agreement import assert_max_delta, csv_rows, index_rows, run_evidence


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    result_dir, paper = run_evidence("DWT", tmp_path_factory)
    quality = index_rows(csv_rows(result_dir, "quality.csv"), ("image",))
    attacks = index_rows(csv_rows(result_dir, "summary.csv"), ("image", "attack"))
    assert {image for image, in quality} == set(paper["mssim_table7_text"])
    expected = {(image, attack) for table in ("table7", "table8")
                for image, values in paper[table].items() for attack in values["NCQM"]}
    assert expected <= set(attacks)
    return paper, quality, attacks


def test_quality_psnr_42_db(evidence):
    """Four article hosts are calibrated to 42 dB."""
    _, quality, _ = evidence
    assert_max_delta(((image, row["psnr_db"], 42.0) for (image,), row in quality.items()),
                     0.001, "DWT clean PSNR (dB)")


def test_quality_mssim(evidence):
    """All four published MSSIM values."""
    paper, quality, _ = evidence
    assert_max_delta(((image, quality[(image,)]["mssim"], reference)
                      for image, reference in paper["mssim_table7_text"].items()),
                     0.0007, "DWT MSSIM")


@pytest.mark.parametrize("table,limit", [("table7", 2.35), ("table8", 1.95)])
def test_ber_tables(evidence, table, limit):
    """Compare every NCQM BER with the run in percentage points, not BER fractions."""
    paper, _, attacks = evidence
    assert_max_delta(((f"{image}/{attack}", 100 * float(attacks[image, attack]["ber_mean"]), reference)
                      for image, values in paper[table].items()
                      for attack, reference in values["NCQM"].items()),
                     limit, f"DWT {table} BER (percentage points)")
