"""Build frequency algorithms and compare runs with YAML paper protocols."""

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data_embed"
_BACKEND_DIR = None
_TEMP_BUILD = None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spec(path):
    """Load one standalone experiment protocol and its published values."""
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(spec, dict) and spec.get("schema_version") == 1, f"invalid paper protocol: {path}"
    assert spec["experiment"]["algorithm"] in ("DCT", "DWT", "DFT"), f"unsupported paper protocol: {path}"
    return spec


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


def run_evidence(spec_path, tmp_path_factory):
    """Run the article protocol without loading any report generator."""
    spec_path = Path(spec_path).resolve()
    spec = load_spec(spec_path)
    experiment = spec["experiment"]
    algorithm = experiment["algorithm"]
    backend_dir = build_backend()
    result_dir = tmp_path_factory.mktemp(f"{algorithm.lower()}_paper_run")
    assert not any(result_dir.iterdir()), f"experiment directory is not empty: {result_dir}"
    command = [
        sys.executable,
        "-u",
        str(DATA / "paper_experiment_runner.py"),
        "--config",
        str(spec_path),
        "--output-dir",
        str(result_dir),
        "--backend-dir",
        str(backend_dir),
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
    expected = set(experiment["outputs"])
    assert {path.name for path in result_dir.iterdir()} == expected
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_kind"] == f"test_{algorithm}"
    assert manifest["seed"] == experiment["seed"]
    assert manifest["repeats"] == experiment["repeats"]
    assert Path(manifest["backend_dir"]).resolve() == backend_dir.resolve()
    source = ROOT / experiment["source"]
    assert sha256(source) == manifest["source_sha256"]
    assert sha256(spec_path) == manifest["config_sha256"]
    assert set(manifest["inputs"]) == set(experiment["hosts"])
    for name, info in manifest["inputs"].items():
        expected_host = experiment["hosts"][name]
        assert info["file"] == expected_host["file"]
        host = ROOT / experiment["images_dir"] / info["file"]
        assert sha256(host) == expected_host["sha256"] == info["sha256"], f"host image changed: {name}"
    watermark = experiment["watermark"]
    if "file" in watermark:
        assert sha256(ROOT / watermark["file"]) == watermark["sha256"] == manifest["watermark_sha256"]
    return result_dir, spec


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
