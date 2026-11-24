"""Subprocess coverage for the public ``main.py`` pipeline CLI."""

import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = REPO_ROOT / "main.py"
COMMAND_TIMEOUT = 45


def _run_cli(tmp_path, *arguments):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO_ROOT), env.get("PYTHONPATH"))))
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        [sys.executable, str(MAIN_PY), *map(str, arguments)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name != "nt",
        creationflags=creationflags,
    )
    try:
        stdout, stderr = process.communicate(timeout=COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=10,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _write_config(
    tmp_path,
    experiment_data,
    *,
    workers=1,
    embedding="ADC_ABW",
    overwrite=True,
    parameterization=None,
    watermark_scope=None,
):
    images = Path(experiment_data["dataset"]["path"])
    images.mkdir(parents=True)
    for index in range(2):
        pixels = [(x * 7 + y * 3 + index * 11) % 256 for y in range(128) for x in range(128)]
        image = Image.new("L", (128, 128))
        image.putdata(pixels)
        image.save(images / f"sample-{index}.png")

    experiment_data["execution"].update(workers=workers, max_in_flight=workers)
    experiment_data["output"]["overwrite"] = overwrite
    embedding_config = experiment_data["embeddings"][0]
    embedding_config["name"] = embedding
    embedding_params = embedding_config["embedding"]["params"]
    if parameterization:
        fixed = embedding_params["fixed"]
        fixed.pop("attack_angle")
        if parameterization == "grid":
            embedding_params["grid"] = {"attack_angle": [25.0, 30.0]}
        elif parameterization == "variants":
            embedding_params["variants"] = [{"attack_angle": angle} for angle in (25.0, 30.0)]
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")
    if watermark_scope:
        experiment_data["watermark"] = {"type": "random_bits", "length": 4, "scope": watermark_scope}
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(experiment_data, sort_keys=False), encoding="utf-8")
    return config_path


def _database_rows(database_path):
    with closing(sqlite3.connect(database_path)) as connection:
        return connection.execute("SELECT status, result_json, error FROM cases ORDER BY case_id").fetchall()


def test_help_and_unknown_arguments(tmp_path):
    help_result = _run_cli(tmp_path, "--help")
    assert help_result.returncode == 0
    assert "--config" in help_result.stdout
    assert "--validate-only" in help_result.stdout

    invalid_result = _run_cli(tmp_path, "--not-a-real-option")
    assert invalid_result.returncode == 2


@pytest.mark.parametrize(
    "contents",
    ["not: [valid", "schema_version: 1\nunknown_field: true\n"],
    ids=["malformed-yaml", "invalid-schema"],
)
def test_invalid_config_exits_two(tmp_path, contents):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(contents, encoding="utf-8")

    result = _run_cli(tmp_path, "--config", config_path)

    assert result.returncode == 2


def test_missing_config_exits_two(tmp_path):
    result = _run_cli(tmp_path, "--config", tmp_path / "missing.yaml")

    assert result.returncode == 2


def test_serial_backend_is_rejected_before_output_creation(tmp_path, experiment_data):
    config_path = tmp_path / "serial.yaml"
    output_path = Path(experiment_data["output"]["directory"])
    experiment_data["execution"].update(backend="serial", workers=1, max_in_flight=1)
    config_path.write_text(yaml.safe_dump(experiment_data, sort_keys=False), encoding="utf-8")

    result = _run_cli(tmp_path, "--config", config_path)

    assert result.returncode == 2
    assert not output_path.exists()


def test_validate_only_does_not_require_dataset_or_create_output(tmp_path, experiment_data):
    config_path = tmp_path / "experiment.yaml"
    output_path = Path(experiment_data["output"]["directory"])
    config_path.write_text(yaml.safe_dump(experiment_data, sort_keys=False), encoding="utf-8")

    result = _run_cli(tmp_path, "--config", config_path, "--validate-only")

    assert result.returncode == 0, result.stderr
    assert "valid" in result.stdout.lower()
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("workers", "expected_cases", "config_flag", "parameterization", "watermark_scope"),
    [
        (1, 4, "--config", None, None),
        (2, 8, "-c", "grid", None),
        (1, 8, "--config", "variants", None),
        (1, 4, "-c", None, "per_image"),
    ],
    ids=["fixed", "grid", "linked-variants", "random-per-image"],
)
def test_live_pipeline_persists_metrics_and_resumes(
    tmp_path, experiment_data, workers, expected_cases, config_flag, parameterization, watermark_scope
):
    config_path = _write_config(
        tmp_path,
        experiment_data,
        workers=workers,
        parameterization=parameterization,
        watermark_scope=watermark_scope,
    )
    output_path = Path(experiment_data["output"]["directory"])
    database_path = output_path / "results.sqlite3"

    result = _run_cli(tmp_path, config_flag, config_path)

    assert result.returncode == 0, result.stderr
    assert f"Completed {expected_cases} cases." in result.stdout
    rows = _database_rows(database_path)
    assert len(rows) == expected_cases
    assert all(status == "done" and error is None for status, _, error in rows)
    with closing(sqlite3.connect(database_path)) as connection:
        known_paths = {row[0] for row in connection.execute("SELECT path_id FROM operation_paths")}

    clean_metrics = 0
    for status, result_json, _ in rows:
        assert status == "done"
        metrics = json.loads(result_json)["metrics"]
        assert set(metrics) == {"embedding-mse", "attack-mse", "ber"}

        for entry in metrics.values():
            assert set(entry) == {"value", "path_id"}
            assert entry["path_id"] in known_paths
            assert math.isfinite(entry["value"]) and entry["value"] >= 0

        assert 0 <= metrics["ber"]["value"] <= 1
        if metrics["attack-mse"]["value"] == 0 and metrics["ber"]["value"] == 0:
            clean_metrics += 1

    assert clean_metrics >= expected_cases // 2
    # clean_metrics = 0
    # for status, result_json, _ in rows:
    #     assert status == "done"
    #     metrics = json.loads(result_json)["metrics"]
    #     assert set(metrics) == {"embedding-mse", "attack-mse", "ber"}
    #     assert all(math.isfinite(value) and value >= 0 for value in metrics.values())
    #     assert 0 <= metrics["ber"] <= 1
    #     if metrics["attack-mse"] == 0 and metrics["ber"] == 0:
    #         clean_metrics += 1
    # assert clean_metrics >= expected_cases // 2

    # Resume without overwrite: completed cases remain unchanged in the database.
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["output"]["overwrite"] = False
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    before = _database_rows(database_path)
    resumed = _run_cli(tmp_path, config_flag, config_path)

    assert resumed.returncode == 0, resumed.stderr
    assert _database_rows(database_path) == before


def test_runtime_case_failures_exit_nonzero_and_are_recorded(tmp_path, experiment_data):
    config_path = _write_config(tmp_path, experiment_data, embedding="MissingEmbedding")

    result = _run_cli(tmp_path, "--config", config_path)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "failed" in (result.stdout + result.stderr).lower()
    rows = _database_rows(Path(experiment_data["output"]["directory"]) / "results.sqlite3")
    assert len(rows) == 4
    assert all(status == "failed" and error for status, _, error in rows)


def test_resume_ignores_unrelated_historical_failure(tmp_path, experiment_data):
    config_path = _write_config(tmp_path, experiment_data)
    database_path = Path(experiment_data["output"]["directory"]) / "results.sqlite3"
    baseline = _run_cli(tmp_path, "--config", config_path)
    assert baseline.returncode == 0, baseline.stderr
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute(
            "INSERT INTO cases (case_id, image_id, status, error) VALUES (?, ?, 'failed', ?)",
            ("unrelated-historical-case", "unrelated-image", "old failure"),
        )
        connection.commit()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["output"]["overwrite"] = False
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    resumed = _run_cli(tmp_path, "--config", config_path)

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "Completed 4 cases." in resumed.stdout
    assert len(_database_rows(database_path)) == 5


def test_resume_with_no_selected_images_does_not_report_historical_success(tmp_path, experiment_data):
    config_path = _write_config(tmp_path, experiment_data)
    database_path = Path(experiment_data["output"]["directory"]) / "results.sqlite3"
    baseline = _run_cli(tmp_path, "--config", config_path)
    assert baseline.returncode == 0, baseline.stderr

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["output"]["overwrite"] = False
    config["dataset"]["include"] = ["absent-*.png"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    resumed = _run_cli(tmp_path, "--config", config_path)

    assert resumed.returncode == 1
    assert "Completed 4 cases." not in resumed.stdout
    assert len(_database_rows(database_path)) == 4
