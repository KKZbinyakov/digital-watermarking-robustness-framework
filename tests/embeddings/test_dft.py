"""Recompute DFT article experiments and compare published curves."""

import math
from pathlib import Path

import pytest

from ._paper_support import assert_max_delta, csv_rows, finite, index_rows, run_evidence

SPEC = Path(__file__).resolve().parents[2] / "examples" / "paper_verification" / "dft.yaml"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    result_dir, spec = run_evidence(SPEC, tmp_path_factory)
    paper, agreement = spec["reference"], spec["agreement"]
    quality = index_rows(csv_rows(result_dir, "quality.csv"), ("image", "trial"))
    weibull = index_rows(csv_rows(result_dir, "weibull.csv"), ("region",))
    summary = index_rows(csv_rows(result_dir, "summary.csv"), ("image", "attack", "parameter"))
    experiment = spec["experiment"]
    assert len(quality) == experiment["repeats"] * len(experiment["hosts"])
    assert set(weibull) == {(str(region),) for region in range(1, experiment["weibull"]["regions"] + 1)}
    median = agreement["median3_text_ber"]
    assert ("ALL", median["attack"], str(median["parameter"])) in summary
    return paper, agreement, quality, weibull, summary


def test_clean_psnr_45_db(evidence):
    """Every one of 192 runs targets the paper's 45 dB."""
    _, agreement, quality, _, _ = evidence
    target = agreement["clean_psnr_db"]
    assert_max_delta(
        ((f"{image}/trial={trial}", row["psnr_db"], target["target"]) for (image, trial), row in quality.items()),
        target["limit"],
        "DFT clean PSNR (dB)",
    )


@pytest.mark.parametrize("parameter", ("alpha", "beta"))
def test_table1_weibull(evidence, parameter):
    """All 16 article/run points for each Weibull parameter; alpha uses article scaling."""
    paper, agreement, _, weibull, _ = evidence
    assert_max_delta(
        (
            (f"region {region}", weibull[str(region),][f"measured_{parameter}"], reference)
            for region, reference in enumerate(paper["table1"][parameter], start=1)
        ),
        agreement["weibull"][parameter],
        f"DFT Weibull {parameter}",
    )


def _paper_log_ber(bpp, xs, ys):
    """Interpolate the digitized paper line in log-log coordinates."""
    for left in range(len(xs) - 1):
        if xs[left] <= bpp <= xs[left + 1]:
            weight = (math.log10(bpp) - math.log10(xs[left])) / (math.log10(xs[left + 1]) - math.log10(xs[left]))
            return math.log10(ys[left]) + weight * (math.log10(ys[left + 1]) - math.log10(ys[left]))
    raise AssertionError(f"bpp outside published Figure 3: {bpp}")


@pytest.mark.parametrize("codec", ("jpeg", "jpeg2000"))
def test_figure3_compression_ber(evidence, codec):
    """Every run point inside paper range."""
    paper, agreement, _, _, summary = evidence
    bounds = agreement["figure3_log10_ber"][codec]
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
        pairs.append(
            (f"{codec}/quality={parameter}/bpp={bpp:.5f}", math.log10(display_ber), _paper_log_ber(bpp, xs, ys))
        )
    assert len(pairs) == bounds["point_count"], (
        f"Figure 3 {codec}: compared {len(pairs)} points, expected {bounds['point_count']}"
    )
    assert_max_delta(pairs, bounds["limit"], f"DFT Figure 3 {codec} log10 BER")


def test_median3_text_ber(evidence):
    """Paper text BER vs run."""
    paper, agreement, _, _, summary = evidence
    bounds = agreement["median3_text_ber"]
    assert_max_delta(
        (("3x3 median", summary["ALL", bounds["attack"], str(bounds["parameter"])]["ber"], paper["median3_text_ber"]),),
        bounds["limit"],
        "DFT median 3x3 BER",
    )
