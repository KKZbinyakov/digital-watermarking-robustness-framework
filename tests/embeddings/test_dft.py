"""Recompute DFT article experiments and compare published curves."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_agreement import assert_max_delta, csv_rows, finite, index_rows, run_evidence


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    result_dir, paper = run_evidence("DFT", tmp_path_factory)
    quality = index_rows(csv_rows(result_dir, "quality.csv"), ("image", "trial"))
    weibull = index_rows(csv_rows(result_dir, "weibull.csv"), ("region",))
    summary = index_rows(csv_rows(result_dir, "summary.csv"), ("image", "attack", "parameter"))
    assert len(quality) == 192
    assert set(weibull) == {(str(region),) for region in range(1, 17)}
    assert ("ALL", "median", "3") in summary
    return paper, quality, weibull, summary


def test_clean_psnr_45_db(evidence):
    """Every one of 192 runs targets the paper's 45 dB."""
    _, quality, _, _ = evidence
    assert_max_delta(((f"{image}/trial={trial}", row["psnr_db"], 45.0)
                      for (image, trial), row in quality.items()),
                     0.001, "DFT clean PSNR (dB)")


@pytest.mark.parametrize("parameter,limit", [("alpha", 0.023), ("beta", 0.29)])
def test_table1_weibull(evidence, parameter, limit):
    """All 16 article/run points for each Weibull parameter; alpha uses article scaling."""
    paper, _, weibull, _ = evidence
    assert_max_delta(((f"region {region}", weibull[str(region),][f"measured_{parameter}"], reference)
                      for region, reference in enumerate(paper["table1"][parameter], start=1)),
                     limit, f"DFT Weibull {parameter}")


def _paper_log_ber(bpp, xs, ys):
    """Interpolate the digitized paper line in log-log coordinates."""
    for left in range(len(xs) - 1):
        if xs[left] <= bpp <= xs[left + 1]:
            weight = (math.log10(bpp) - math.log10(xs[left])) / (math.log10(xs[left + 1]) - math.log10(xs[left]))
            return math.log10(ys[left]) + weight * (math.log10(ys[left + 1]) - math.log10(ys[left]))
    raise AssertionError(f"bpp outside published Figure 3: {bpp}")


@pytest.mark.parametrize("codec,limit,count", [("jpeg", 0.50, 9), ("jpeg2000", 1.48, 8)])
def test_figure3_compression_ber(evidence, codec, limit, count):
    """Every run point inside paper range."""
    paper, _, _, summary = evidence
    figure = paper["figure3"]
    xs = figure["bpp_approx"]
    ys = figure[codec + "_ber_approx"]
    pairs = []
    for (image, attack, parameter), row in summary.items():
        if image != "ALL" or attack != codec:
            continue
        bpp = finite(row["bpp_mean"])
        if not xs[0] <= bpp <= xs[-1]:
            continue
        ber = finite(row["ber"])
        assert 0 <= ber <= 1
        display_ber = max(ber, 1 / int(row["total_bits"]))
        pairs.append((f"{codec}/quality={parameter}/bpp={bpp:.5f}",
                      math.log10(display_ber), _paper_log_ber(bpp, xs, ys)))
    assert len(pairs) == count, f"Figure 3 {codec}: compared {len(pairs)} points, expected {count}"
    assert_max_delta(pairs, limit, f"DFT Figure 3 {codec} log10 BER")


def test_median3_text_ber(evidence):
    """Paper text BER vs run."""
    paper, _, _, summary = evidence
    assert_max_delta((("3x3 median", summary["ALL", "median", "3"]["ber"],
                       paper["median3_text_ber"]),), 0.00052, "DFT median 3x3 BER")
