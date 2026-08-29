import json
import pickle
from pathlib import Path

import pytest
from pydantic import ValidationError

from dwarf.pipeline import (
    ConfigLoadError,
    ExperimentConfig,
    ParameterSpace,
    load_experiment_config,
)


def valid_config_data(tmp_path: Path) -> dict:
    return {
        "schema_version": 1,
        "experiment": {
            "name": "DCT JPEG smoke test",
            "seed": 42,
            "repeats": 2,
            "max_cases": 1000,
            "tags": ["smoke", "frequency"],
        },
        "plugins": ["custom_solutions.attacks"],
        "dataset": {
            "type": "directory",
            "path": str(tmp_path / "images"),
            "recursive": True,
            "extensions": [".PNG", ".jpg"],
            "preprocessing": {
                "color_mode": "RGB",
                "resize": [512, 512],
                "resample": "lanczos",
                "exif_transpose": True,
            },
        },
        "watermark": {
            "type": "random_bits",
            "length": 64,
            "scope": "fixed_for_run",
        },
        "embeddings": [
            {
                "id": "dct",
                "name": "DCT",
                "embedding": {
                    "params": {
                        "fixed": {"threshold": 25.0},
                        "grid": {"margin": [100.0, 150.0]},
                    }
                },
                "extraction": {
                    "params": {"fixed": {"threshold": 25.0}}
                },
            }
        ],
        "attack_scenarios": [
            {"id": "clean", "steps": []},
            {
                "id": "jpeg",
                "steps": [
                    {
                        "name": "Jpeg",
                        "params": {"grid": {"quality": [30, 75, 90]}},
                    }
                ],
            },
        ],
        "metrics": [
            {
                "id": "embedding-psnr",
                "name": "PSNR",
                "checkpoint": "after_embedding",
                "inputs": {
                    "original_image": "original_image",
                    "distorted_image": "embedded_image",
                },
            },
            {
                "id": "watermark-ber",
                "name": "BER",
                "checkpoint": "after_extraction",
                "inputs": {
                    "original_bits": "original_watermark",
                    "extracted_bits": "extracted_watermark",
                },
                "params": {"fixed": {"allow_length_mismatch": False}},
            },
        ],
        "execution": {
            "backend": "process",
            "workers": 4,
            "max_in_flight": 8,
            "start_method": "spawn",
            "fail_fast": False,
            "show_progress": True,
        },
        "output": {
            "directory": str(tmp_path / "runs"),
            "database": "sqlite",
            "database_filename": "results.sqlite3",
            "save_images": "sampled",
            "sample_fraction": 0.1,
            "exports": ["csv", "jsonl"],
            "overwrite": False,
        },
        "report": {
            "enabled": True,
            "formats": ["csv", "html"],
            "group_by": ["embedding", "attack_scenario", "metric"],
            "include_failures": True,
            "float_precision": 6,
        },
    }


def assert_error_type(error: ValidationError, expected_type: str) -> None:
    assert any(item["type"] == expected_type for item in error.errors())


def test_programmatic_configuration_is_valid_and_immutable(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(valid_config_data(tmp_path))

    assert config.schema_version == 1
    assert config.dataset.extensions == (".png", ".jpg")
    assert config.embeddings[0].embedding.params.declared_variant_count == 2
    assert config.attack_scenarios[1].steps[0].params.declared_variant_count == 3

    with pytest.raises(ValidationError, match="frozen"):
        config.experiment.name = "changed"

    with pytest.raises(TypeError, match="immutable"):
        config.embeddings[0].embedding.params.fixed["threshold"] = 30.0

    restored = pickle.loads(pickle.dumps(config))
    assert restored == config


def test_validation_detaches_nested_parameter_values(tmp_path: Path) -> None:
    data = valid_config_data(tmp_path)
    kernel = [3, {"shape": [5, 5]}]
    data["embeddings"][0]["embedding"]["params"]["fixed"]["kernel"] = kernel

    config = ExperimentConfig.model_validate(data)
    kernel[1]["shape"][0] = 99

    stored = config.embeddings[0].embedding.params.fixed["kernel"]
    assert stored == (3, {"shape": (5, 5)})


def test_minimal_configuration_adds_clean_baseline() -> None:
    config = ExperimentConfig.model_validate(
        {
            "schema_version": 1,
            "experiment": {"name": "minimal"},
            "dataset": {"type": "directory", "path": "images"},
            "watermark": {"type": "fixed_bits", "bits": "0101"},
            "embeddings": [{"id": "dct", "name": "DCT"}],
        }
    )

    assert tuple(item.id for item in config.attack_scenarios) == ("clean",)
    assert config.execution.backend == "serial"
    assert config.output.database_filename == "results.sqlite3"
    assert config.report.formats == ("csv",)


def test_model_dump_is_json_serialisable(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(valid_config_data(tmp_path))

    dumped = config.model_dump(mode="json")

    assert dumped["embeddings"][0]["embedding"]["params"]["grid"]["margin"] == [
        100.0,
        150.0,
    ]
    json.dumps(dumped, allow_nan=False)


def test_json_schema_forbids_extra_fields() -> None:
    schema = ExperimentConfig.model_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == 1


def test_yaml_and_python_configs_produce_equal_models(tmp_path: Path) -> None:
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text(
        """
schema_version: 1
experiment:
  name: equivalent
  seed: 7
dataset:
  type: directory
  path: data/images
watermark:
  type: random_bits
  length: 16
embeddings:
  - id: dct
    name: DCT
    embedding:
      params:
        fixed:
          kernel: [3, 3]
        grid:
          margin: [100.0, 150.0]
output:
  directory: output
""".lstrip(),
        encoding="utf-8",
    )

    from_yaml = load_experiment_config(str(config_file))
    from_python = ExperimentConfig.model_validate(
        {
            "schema_version": 1,
            "experiment": {"name": "equivalent", "seed": 7},
            "dataset": {"type": "directory", "path": "data/images"},
            "watermark": {"type": "random_bits", "length": 16},
            "embeddings": [
                {
                    "id": "dct",
                    "name": "DCT",
                    "embedding": {
                        "params": {
                            "fixed": {"kernel": (3, 3)},
                            "grid": {"margin": [100.0, 150.0]},
                        }
                    },
                }
            ],
            "output": {"directory": "output"},
        }
    ).resolve_paths(str(tmp_path))

    assert from_yaml == from_python
    assert from_yaml.dataset.path == (tmp_path / "data/images").resolve()
    assert from_yaml.output.directory == (tmp_path / "output").resolve()
    assert from_yaml.embeddings[0].embedding.params.fixed["kernel"] == (3, 3)


def test_yaml_loader_can_keep_relative_paths(tmp_path: Path) -> None:
    config_file = tmp_path / "experiment.yml"
    config_file.write_text(
        """
schema_version: 1
experiment: {name: relative}
dataset: {type: directory, path: images}
watermark: {type: fixed_bits, bits: "01"}
embeddings: [{id: dct, name: DCT}]
""".lstrip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_file, resolve_paths=False)

    assert config.dataset.path == Path("images")
    assert config.output.directory == Path("runs")


@pytest.mark.parametrize(
    ("mutator", "expected_type", "message"),
    [
        (
            lambda data: data.update({"unknown": 1}),
            "extra_forbidden",
            None,
        ),
        (
            lambda data: data["dataset"].update({"recusrive": True}),
            "extra_forbidden",
            None,
        ),
        (
            lambda data: data.update({"schema_version": 2}),
            "value_error",
            "unsupported schema_version",
        ),
        (
            lambda data: data.update({"schema_version": "1"}),
            "value_error",
            "integer 1",
        ),
        (
            lambda data: data["experiment"].update({"repeats": "2"}),
            "int_type",
            None,
        ),
        (
            lambda data: data["experiment"].update({"repeats": True}),
            "int_type",
            None,
        ),
        (
            lambda data: data["dataset"].update({"recursive": "false"}),
            "bool_type",
            None,
        ),
        (
            lambda data: data["execution"].update({"workers": "4"}),
            "int_type",
            None,
        ),
    ],
)
def test_strict_schema_rejects_typos_and_coercion(
    tmp_path: Path,
    mutator,
    expected_type: str,
    message: str,
) -> None:
    data = valid_config_data(tmp_path)
    mutator(data)

    with pytest.raises(ValidationError) as caught:
        ExperimentConfig.model_validate(data)

    assert_error_type(caught.value, expected_type)
    if message is not None:
        assert message in str(caught.value)


def test_parameter_space_rejects_grid_and_variants_together() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        ParameterSpace.model_validate(
            {
                "grid": {"quality": [50, 75]},
                "variants": [{"quality": 90}],
            }
        )


def test_parameter_space_rejects_fixed_variable_overlap() -> None:
    with pytest.raises(ValidationError, match="both fixed and variable"):
        ParameterSpace.model_validate(
            {
                "fixed": {"quality": 75},
                "grid": {"quality": [50, 90]},
            }
        )


def test_parameter_space_rejects_empty_and_duplicate_grid_candidates() -> None:
    with pytest.raises(ValidationError, match="at least one candidate"):
        ParameterSpace.model_validate({"grid": {"quality": []}})

    with pytest.raises(ValidationError, match="duplicate candidates"):
        ParameterSpace.model_validate({"grid": {"quality": [75, 75]}})


def test_parameter_space_rejects_empty_explicit_variants() -> None:
    with pytest.raises(ValidationError, match="at least one parameter combination"):
        ParameterSpace.model_validate({"variants": []})


def test_parameter_space_rejects_mismatched_and_duplicate_variants() -> None:
    with pytest.raises(ValidationError, match="same parameter keys"):
        ParameterSpace.model_validate(
            {
                "variants": [
                    {"quality": 50, "method": "a"},
                    {"quality": 75},
                ]
            }
        )

    with pytest.raises(ValidationError, match="duplicate parameter combinations"):
        ParameterSpace.model_validate(
            {"variants": [{"quality": 50}, {"quality": 50}]}
        )


def test_parameter_space_rejects_reserved_runtime_arguments() -> None:
    with pytest.raises(ValidationError, match="runtime-managed"):
        ParameterSpace.model_validate({"fixed": {"input_image": "image.png"}})


def test_parameter_space_rejects_invalid_parameter_names() -> None:
    with pytest.raises(ValidationError) as caught:
        ParameterSpace.model_validate({"fixed": {"bad-name": 1}})

    assert_error_type(caught.value, "string_pattern_mismatch")


def test_parameter_space_rejects_non_json_and_non_finite_values() -> None:
    with pytest.raises(ValidationError, match="only JSON-compatible"):
        ParameterSpace.model_validate({"fixed": {"callback": object()}})

    with pytest.raises(ValidationError, match="NaN or infinity"):
        ParameterSpace.model_validate({"fixed": {"threshold": float("nan")}})

    with pytest.raises(ValidationError, match="non-string mapping key"):
        ParameterSpace.model_validate({"fixed": {"options": {1: "value"}}})


def test_parameter_space_normalises_nested_containers_and_counts_variants() -> None:
    parameters = ParameterSpace.model_validate(
        {
            "fixed": {"kernel": [3, 3]},
            "grid": {
                "method": ["a", "b"],
                "options": [({"window": [5, 5]},), ({"window": [7, 7]},)],
            },
        }
    )

    assert parameters.fixed == {"kernel": (3, 3)}
    assert parameters.grid["options"][0] == ({"window": (5, 5)},)
    assert parameters.declared_variant_count == 4

    variants = ParameterSpace.model_validate(
        {"variants": [{"quality": 50}, {"quality": 75}]}
    )
    assert variants.declared_variant_count == 2


def test_fixed_watermark_only_accepts_binary_string() -> None:
    data = {
        "schema_version": 1,
        "experiment": {"name": "invalid bits"},
        "dataset": {"type": "directory", "path": "images"},
        "watermark": {"type": "fixed_bits", "bits": "0102"},
        "embeddings": [{"id": "dct", "name": "DCT"}],
    }

    with pytest.raises(ValidationError) as caught:
        ExperimentConfig.model_validate(data)

    assert_error_type(caught.value, "string_pattern_mismatch")


@pytest.mark.parametrize("collection", ["embeddings", "attack_scenarios", "metrics"])
def test_duplicate_component_ids_are_rejected(
    tmp_path: Path,
    collection: str,
) -> None:
    data = valid_config_data(tmp_path)
    if collection == "embeddings":
        data[collection].append({"id": "dct", "name": "DFT"})
    elif collection == "attack_scenarios":
        data[collection].append({"id": "clean", "steps": []})
    else:
        data[collection].append(
            {
                "id": "embedding-psnr",
                "name": "SSIM",
                "checkpoint": "after_embedding",
                "inputs": {"image": "embedded_image"},
            }
        )

    with pytest.raises(ValidationError, match="duplicate values"):
        ExperimentConfig.model_validate(data)


def test_empty_embedding_and_attack_collections_are_rejected(tmp_path: Path) -> None:
    data = valid_config_data(tmp_path)
    data["embeddings"] = []
    with pytest.raises(ValidationError) as caught:
        ExperimentConfig.model_validate(data)
    assert_error_type(caught.value, "too_short")

    data = valid_config_data(tmp_path)
    data["attack_scenarios"] = []
    with pytest.raises(ValidationError) as caught:
        ExperimentConfig.model_validate(data)
    assert_error_type(caught.value, "too_short")


def test_metric_cannot_read_future_artifact(tmp_path: Path) -> None:
    data = valid_config_data(tmp_path)
    data["metrics"][0]["inputs"]["future"] = "attacked_image"

    with pytest.raises(ValidationError, match="cannot access artifacts"):
        ExperimentConfig.model_validate(data)


def test_metric_inputs_and_parameters_cannot_collide(tmp_path: Path) -> None:
    data = valid_config_data(tmp_path)
    data["metrics"][0]["inputs"]["image"] = "original_image"
    data["metrics"][0]["params"] = {"fixed": {"image": "rgb"}}

    with pytest.raises(ValidationError, match="distinct argument names"):
        ExperimentConfig.model_validate(data)


def test_metric_inputs_must_not_be_empty(tmp_path: Path) -> None:
    data = valid_config_data(tmp_path)
    data["metrics"][0]["inputs"] = {}

    with pytest.raises(ValidationError, match="must not be empty"):
        ExperimentConfig.model_validate(data)


def test_serial_execution_rejects_parallel_only_settings(tmp_path: Path) -> None:
    data = valid_config_data(tmp_path)
    data["execution"] = {
        "backend": "serial",
        "workers": 2,
        "max_in_flight": 2,
    }

    with pytest.raises(ValidationError, match="serial execution requires workers"):
        ExperimentConfig.model_validate(data)

    data["execution"] = {
        "backend": "serial",
        "workers": 1,
        "max_in_flight": 1,
        "start_method": "spawn",
    }
    with pytest.raises(ValidationError, match="only applicable to process"):
        ExperimentConfig.model_validate(data)


def test_process_execution_rejects_too_small_in_flight_limit(tmp_path: Path) -> None:
    data = valid_config_data(tmp_path)
    data["execution"]["workers"] = 8
    data["execution"]["max_in_flight"] = 4

    with pytest.raises(ValidationError, match="greater than or equal to workers"):
        ExperimentConfig.model_validate(data)


def test_sample_fraction_is_strict_and_only_valid_for_sampling(tmp_path: Path) -> None:
    data = valid_config_data(tmp_path)
    del data["output"]["sample_fraction"]

    with pytest.raises(ValidationError, match="sample_fraction is required"):
        ExperimentConfig.model_validate(data)

    data = valid_config_data(tmp_path)
    data["output"]["save_images"] = "all"
    with pytest.raises(ValidationError, match="only allowed"):
        ExperimentConfig.model_validate(data)

    data = valid_config_data(tmp_path)
    data["output"]["sample_fraction"] = 1
    with pytest.raises(ValidationError, match="floating-point"):
        ExperimentConfig.model_validate(data)


def test_database_filename_must_not_contain_a_path(tmp_path: Path) -> None:
    for filename in ("sub/results.db", r"sub\results.db", "results.txt"):
        data = valid_config_data(tmp_path)
        data["output"]["database_filename"] = filename
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(data)


def test_dataset_normalises_extensions_and_rejects_ambiguous_patterns(
    tmp_path: Path,
) -> None:
    data = valid_config_data(tmp_path)
    data["dataset"]["extensions"] = [".PNG", ".png"]
    with pytest.raises(ValidationError, match="duplicate values"):
        ExperimentConfig.model_validate(data)

    data = valid_config_data(tmp_path)
    data["dataset"]["include"] = ["**/*.png"]
    data["dataset"]["exclude"] = ["**/*.png"]
    with pytest.raises(ValidationError, match="same patterns"):
        ExperimentConfig.model_validate(data)


def test_dataset_path_is_not_required_to_exist(tmp_path: Path) -> None:
    data = valid_config_data(tmp_path)
    assert not Path(data["dataset"]["path"]).exists()

    config = ExperimentConfig.model_validate(data)

    assert config.dataset.path == Path(data["dataset"]["path"])


def test_duplicate_plugins_are_rejected(tmp_path: Path) -> None:
    data = valid_config_data(tmp_path)
    data["plugins"] = ["custom.attacks", "custom.attacks"]

    with pytest.raises(ValidationError, match="duplicate values"):
        ExperimentConfig.model_validate(data)


def test_yaml_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    config_file = tmp_path / "duplicate.yaml"
    config_file.write_text(
        """
schema_version: 1
schema_version: 1
experiment: {name: duplicate}
dataset: {type: directory, path: images}
watermark: {type: fixed_bits, bits: "01"}
embeddings: [{id: dct, name: DCT}]
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError, match="duplicate key"):
        load_experiment_config(config_file)


def test_yaml_loader_uses_unambiguous_boolean_rules(tmp_path: Path) -> None:
    config_file = tmp_path / "booleans.yaml"
    config_file.write_text(
        """
schema_version: 1
experiment: {name: booleans}
dataset: {type: directory, path: images}
watermark: {type: fixed_bits, bits: "01"}
embeddings:
  - id: dct
    name: DCT
    embedding:
      params:
        fixed:
          mode: on
execution:
  fail_fast: true
""".lstrip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_file, resolve_paths=False)

    assert config.embeddings[0].embedding.params.fixed["mode"] == "on"
    assert config.execution.fail_fast is True


def test_yaml_loader_rejects_legacy_boolean_spelling_for_typed_fields(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "legacy-boolean.yaml"
    config_file.write_text(
        """
schema_version: 1
experiment: {name: booleans}
dataset:
  type: directory
  path: images
  recursive: yes
watermark: {type: fixed_bits, bits: "01"}
embeddings: [{id: dct, name: DCT}]
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError) as caught:
        load_experiment_config(config_file)

    assert_error_type(caught.value, "bool_type")


def test_yaml_loader_rejects_python_specific_tags(tmp_path: Path) -> None:
    config_file = tmp_path / "unsafe.yaml"
    config_file.write_text(
        """
schema_version: 1
experiment: {name: unsafe}
dataset: {type: directory, path: images}
watermark: {type: fixed_bits, bits: "01"}
embeddings:
  - id: dct
    name: DCT
    embedding:
      params:
        fixed:
          callback: !!python/name:os.system
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError, match="invalid YAML"):
        load_experiment_config(config_file)


@pytest.mark.parametrize(
    ("name", "contents", "message"),
    [
        ("empty.yaml", "", "empty"),
        ("sequence.yaml", "- one\n- two\n", "root must be a mapping"),
        ("broken.yaml", "experiment: [\n", "invalid YAML"),
        (
            "multiple.yaml",
            "schema_version: 1\n---\nschema_version: 1\n",
            "invalid YAML",
        ),
    ],
)
def test_yaml_loader_reports_file_level_errors(
    tmp_path: Path,
    name: str,
    contents: str,
    message: str,
) -> None:
    config_file = tmp_path / name
    config_file.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigLoadError, match=message):
        load_experiment_config(config_file)


def test_yaml_loader_rejects_wrong_extension_and_missing_file(tmp_path: Path) -> None:
    wrong_extension = tmp_path / "experiment.json"
    wrong_extension.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match=".yaml or .yml"):
        load_experiment_config(wrong_extension)

    with pytest.raises(ConfigLoadError, match="does not exist"):
        load_experiment_config(tmp_path / "missing.yaml")


def test_yaml_schema_errors_remain_structured_validation_errors(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid-schema.yaml"
    config_file.write_text(
        """
schema_version: 1
experiment: {name: invalid}
dataset: {type: directory, path: images, recusrive: true}
watermark: {type: fixed_bits, bits: "01"}
embeddings: [{id: dct, name: DCT}]
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError) as caught:
        load_experiment_config(config_file)

    assert caught.value.errors()[0]["loc"] == ("dataset", "recusrive")
    assert_error_type(caught.value, "extra_forbidden")
