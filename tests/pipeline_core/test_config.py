"""Public YAML loading and schema validation tests."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dwarf.core.dwarf_exceptions import Dwarf_Exception
from dwarf.pipeline_core.config import ExecutionSpec, ExperimentConfig
from dwarf.pipeline_core.config_loader import load_experiment_config

PIPELINE_EXAMPLES = Path(__file__).parents[2] / "examples" / "pipeline"
EXPERIMENT_CONFIGS = tuple(
    path for path in sorted(PIPELINE_EXAMPLES.glob("*.yaml")) if not path.name.endswith("_reproduction_manifest.yaml")
)
REPRODUCTION_MANIFESTS = tuple(sorted(PIPELINE_EXAMPLES.glob("*_reproduction_manifest.yaml")))


def _write_yaml(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _assert_field_error(path: Path, field_path: tuple[str | int, ...]) -> None:
    with pytest.raises(ValidationError) as caught:
        load_experiment_config(path)

    assert any(error["loc"][: len(field_path)] == field_path for error in caught.value.errors())


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_loads_valid_config_extensions(suffix, tmp_path, experiment_data):
    path = _write_yaml(tmp_path / f"experiment{suffix}", experiment_data)

    config = load_experiment_config(path)

    assert isinstance(config, ExperimentConfig)
    assert config.experiment.name == "pipeline-test"


def test_yaml_12_boolean_words_do_not_coerce_yes_to_true(tmp_path, experiment_data):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(experiment_data, sort_keys=False).replace("enabled: false", "enabled: yes"),
        encoding="utf-8",
    )

    _assert_field_error(path, ("report", "enabled"))


@pytest.mark.parametrize("resolve_paths", [True, False])
def test_relative_paths_resolve_against_config_file(tmp_path, experiment_data, resolve_paths):
    experiment_data["dataset"]["path"] = "images"
    experiment_data["output"]["directory"] = "runs"
    path = _write_yaml(tmp_path / "configs" / "experiment.yaml", experiment_data)

    config = load_experiment_config(path, resolve_paths=resolve_paths)

    expected_dataset = path.parent / "images"
    expected_output = path.parent / "runs"
    if resolve_paths:
        assert config.dataset.path == expected_dataset.resolve()
        assert config.output.directory == expected_output.resolve()
    else:
        assert config.dataset.path == Path("images")
        assert config.output.directory == Path("runs")


@pytest.mark.parametrize(
    ("suffix", "contents", "message"),
    [
        (".json", "{}", r"\.yaml or \.yml"),
        (".yaml", "", "empty"),
        (".yaml", "- one\n- two\n", "root must be a mapping"),
        (".yaml", "schema_version: [\n", "invalid YAML"),
        (".yaml", "a: 1\n---\nb: 2\n", "invalid YAML"),
        (".yaml", "experiment:\n  name: first\n  name: second\n", "duplicate key"),
        (".yaml", "value: !!python/object/apply:os.system ['echo unsafe']\n", "invalid YAML"),
    ],
)
def test_loader_rejects_invalid_yaml_inputs(tmp_path, suffix, contents, message):
    path = tmp_path / f"input{suffix}"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(Dwarf_Exception, match=message):
        load_experiment_config(path)


def test_loader_rejects_missing_file(tmp_path):
    with pytest.raises(Dwarf_Exception, match="does not exist"):
        load_experiment_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    ("mutate", "field_path"),
    [
        (lambda data: data.update(schema_version=2), ("schema_version",)),
        (lambda data: data["experiment"].update(unexpected=True), ("experiment", "unexpected")),
        (lambda data: data["experiment"].update(repeats=0), ("experiment", "repeats")),
        (lambda data: data["execution"].update(workers=0), ("execution", "workers")),
        (
            lambda data: data["embeddings"][0]["embedding"]["params"].update(grid={"strength": []}),
            ("embeddings", 0, "embedding", "params", "grid"),
        ),
        (
            lambda data: data["embeddings"][0]["embedding"]["params"].update(variants=[]),
            ("embeddings", 0, "embedding", "params"),
        ),
        (
            lambda data: data["embeddings"][0]["embedding"]["params"].update(
                grid={"strength": [1, 2]}, variants=[{"mode": "a"}]
            ),
            ("embeddings", 0, "embedding", "params"),
        ),
        (
            lambda data: data["embeddings"][0]["embedding"]["params"]["fixed"].update(input_image="host"),
            ("embeddings", 0, "embedding", "params"),
        ),
        (lambda data: data["watermark"].update(bits="10x"), ("watermark",)),
        (lambda data: data["embeddings"].append(deepcopy(data["embeddings"][0])), ()),
        (lambda data: data["attack_scenarios"].append(deepcopy(data["attack_scenarios"][0])), ()),
        (lambda data: data["metrics"].append(deepcopy(data["metrics"][0])), ()),
        (
            lambda data: data["metrics"][0]["inputs"].update(distorted_image="attacked_image"),
            ("metrics", 0),
        ),
    ],
)
def test_schema_rejects_invalid_values_with_field_locations(tmp_path, experiment_data, mutate, field_path):
    data = deepcopy(experiment_data)
    mutate(data)
    path = _write_yaml(tmp_path / "invalid.yaml", data)

    if field_path:
        _assert_field_error(path, field_path)
    else:
        with pytest.raises(ValidationError):
            load_experiment_config(path)


def test_execution_spec_rejects_inconsistent_serial_settings():
    with pytest.raises(ValidationError):
        ExecutionSpec(backend="serial", workers=2)


@pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda path: path.name)
def test_bundled_experiment_examples_validate_without_running_them(path):
    config = load_experiment_config(path)

    assert isinstance(config, ExperimentConfig)


@pytest.mark.parametrize("path", REPRODUCTION_MANIFESTS, ids=lambda path: path.name)
def test_reproduction_manifests_are_valid_yaml_mappings(path):
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert isinstance(parsed, dict)
