"""Build frequency algorithms and compare independently measured article points."""

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data_embed"
_BACKEND_DIR = None
_TEMP_BUILD = None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_backend():
    """Compile four extensions from current .pyx sources in a temporary directory."""
    global _BACKEND_DIR, _TEMP_BUILD
    if _BACKEND_DIR is not None:
        return _BACKEND_DIR
    _TEMP_BUILD = tempfile.TemporaryDirectory(prefix="frequency_paper_")
    build_root = Path(_TEMP_BUILD.name)
    output = build_root / "lib"
    command = [
        sys.executable,
        str(DATA / "build_frequency_backend.py"),
        "--repo",
        str(ROOT),
        "--output-dir",
        str(output),
    ]
    completed = subprocess.run(command, cwd=build_root, capture_output=True, text=True, timeout=600, check=False)
    assert completed.returncode == 0, (
        f"frequency extension build failed (exit {completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    for name in ("embedding_utils_pyx", "dct", "dwt", "dft"):
        assert len(list(output.rglob(name + "*.pyd" if os.name == "nt" else name + "*.so"))) == 1, (
            f"missing or duplicate compiled extension: {name}"
        )
    _BACKEND_DIR = output
    return output


def run_evidence(algorithm, tmp_path_factory):
    """Run the article protocol without loading any report generator."""
    backend_dir = build_backend()
    result_dir = tmp_path_factory.mktemp(f"{algorithm.lower()}_paper_run")
    assert not any(result_dir.iterdir()), f"experiment directory is not empty: {result_dir}"
    paper_path = DATA / f"{algorithm.lower()}_paper_reference.json"
    repeats = {"DCT": 10, "DWT": 10, "DFT": 64}[algorithm]
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("_paper_experiment_runner.py")),
        "--algorithm",
        algorithm,
        "--output-dir",
        str(result_dir),
        "--backend-dir",
        str(backend_dir),
        "--repeats",
        str(repeats),
        "--seed",
        "2026",
    ]
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=1800, check=False
    )
    assert completed.returncode == 0, (
        f"fresh {algorithm} experiment failed (exit {completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    expected = {
        "DCT": {"table1.csv", "summary.csv", "manifest.json"},
        "DWT": {"quality.csv", "summary.csv", "manifest.json"},
        "DFT": {"quality.csv", "weibull.csv", "summary.csv", "manifest.json"},
    }[algorithm]
    assert {path.name for path in result_dir.iterdir()} == expected
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_kind"] == f"test_{algorithm}"
    assert manifest["seed"] == 2026
    assert manifest["repeats"] == repeats
    assert Path(manifest["backend_dir"]).resolve() == backend_dir.resolve()
    source = ROOT / "dwarf" / "ready_solutions" / "embedding_solutions" / "frequency" / f"{algorithm.lower()}.pyx"
    assert sha256(source) == manifest["source_sha256"]
    assert sha256(paper_path) == manifest["paper_sha256"]
    paper = json.loads(paper_path.read_text(encoding="utf-8"))
    for name, info in manifest["inputs"].items():
        host = DATA / "images" / algorithm / info["file"]
        assert sha256(host) == info["sha256"], f"host image changed: {name}"
    if algorithm == "DCT":
        assert sha256(DATA / "dct_watermark_approx.png") == manifest["watermark_sha256"]
    return result_dir, paper


def csv_rows(result_dir, filename):
    with (result_dir / filename).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert rows, f"empty experiment evidence: {filename}"
    return rows


def index_rows(rows, fields):
    indexed = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        assert key not in indexed, f"duplicate experiment point: {key}"
        indexed[key] = row
    return indexed


def finite(value):
    number = float(value)
    assert math.isfinite(number), f"non-finite experiment value: {value}"
    return number


def assert_max_delta(pairs, limit, metric):
    """A fixed, metric-specific bound applies to each point, never to the mean."""
    pairs = list(pairs)
    assert pairs, f"no comparison points for {metric}"
    errors = [
        (abs(finite(measured) - finite(reference)), label, measured, reference) for label, measured, reference in pairs
    ]
    worst = max(errors)
    assert worst[0] <= limit, (
        f"{metric}: maximum delta {worst[0]:.9g} exceeds {limit:.9g}; "
        f"point={worst[1]}, run={worst[2]}, paper={worst[3]}"
    )
