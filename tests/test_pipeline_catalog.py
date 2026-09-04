"""Tests for registry discovery and semantic experiment validation."""

import copy
import pickle
import sys
from pathlib import Path

import pytest

from dwarf.core.attack_orchestrator.attack_core import Attack_Core
from dwarf.core.embedding_orchestrator.embedding_core import Embedding_Core
from dwarf.core.expertise_orchestrator.expertise_core import Expertise_Core
from dwarf.pipeline import (
    ArtifactInputSpec,
    ArtifactKind,
    DataContract,
    ExperimentConfig,
    SolutionOperationSpec,
    ParameterKind,
    ParameterLinkSpec,
    ParameterSpec,
    SemanticValidationError,
    SolutionCatalog,
    SolutionConflictError,
    SolutionDiscoveryError,
    SolutionKind,
    SolutionSpec,
    load_experiment_config,
)


class DummyEmbedding:
    @staticmethod
    def embedding(**args):
        raise AssertionError("semantic validation must not execute solutions")

    @staticmethod
    def extraction(**args):
        raise AssertionError("semantic validation must not execute solutions")


class DummyAttack:
    @staticmethod
    def attack(**args):
        raise AssertionError("semantic validation must not execute solutions")


class DummyMetric:
    @staticmethod
    def expertise(**args):
        raise AssertionError("semantic validation must not execute solutions")


class UnsupportedAttack:
    @staticmethod
    def attack(**args):
        return args


class BadSignatureAttack:
    @staticmethod
    def attack(input_image):
        return input_image


DUMMY_EMBEDDING_SPEC = SolutionSpec(
    name="DummyEmbedding",
    kind=SolutionKind.EMBEDDING,
    operations={
        "embedding": SolutionOperationSpec(
            entry_point="embedding",
            parameters={
                "strength": ParameterSpec(ParameterKind.NUMBER, default=1.0, minimum=0.0),
                "mode": ParameterSpec(ParameterKind.STRING, required=True),
            },
            runtime_arguments=("input_image", "watermark_bits"),
        ),
        "extraction": SolutionOperationSpec(
            entry_point="extraction",
            parameters={
                "strength": ParameterSpec(ParameterKind.NUMBER, default=1.0, minimum=0.0),
            },
            runtime_arguments=("input_image", "num_bits"),
        ),
    },
    parameter_links=(
        ParameterLinkSpec(
            "embedding",
            "strength",
            "extraction",
            "strength",
        ),
    ),
    contracts={
        "input": DataContract.RGB_UINT8,
        "watermark": DataContract.BINARY_BITS,
    },
)

DUMMY_ATTACK_SPEC = SolutionSpec(
    name="DummyAttack",
    kind=SolutionKind.ATTACK,
    operations={
        "attack": SolutionOperationSpec(
            entry_point="attack",
            parameters={
                "level": ParameterSpec(ParameterKind.INTEGER, default=1, minimum=0, maximum=10),
                "enabled": ParameterSpec(ParameterKind.BOOLEAN, default=True),
            },
            runtime_arguments=("input_image",),
        )
    },
)

DUMMY_METRIC_SPEC = SolutionSpec(
    name="DummyMetric",
    kind=SolutionKind.METRIC,
    operations={
        "expertise": SolutionOperationSpec(
            entry_point="expertise",
            parameters={"normalise": ParameterSpec(ParameterKind.BOOLEAN, default=False)},
            artifact_inputs={
                "left": ArtifactInputSpec(
                    ("original_image", "embedded_image", "attacked_image"),
                    ArtifactKind.IMAGE,
                ),
                "right": ArtifactInputSpec(
                    ("embedded_image", "attacked_image"),
                    ArtifactKind.IMAGE,
                ),
            },
            checkpoints=("after_embedding", "after_attack"),
        )
    },
)

BAD_SIGNATURE_SPEC = SolutionSpec(
    name="BadSignatureAttack",
    kind=SolutionKind.ATTACK,
    operations={"attack": SolutionOperationSpec(entry_point="attack", runtime_arguments=("input_image",))},
)


def minimal_config():
    return {
        "schema_version": 1,
        "experiment": {"name": "catalog-test"},
        "dataset": {"type": "directory", "path": "images"},
        "watermark": {"type": "fixed_bits", "bits": "1010"},
        "embeddings": [
            {
                "id": "dummy",
                "name": "DummyEmbedding",
                "embedding": {
                    "params": {
                        "fixed": {
                            "mode": "robust",
                            "strength": 1.0,
                        }
                    }
                },
                "extraction": {"params": {"fixed": {"strength": 1.0}}},
            }
        ],
        "attack_scenarios": [
            {"id": "clean", "steps": []},
            {
                "id": "attack",
                "steps": [
                    {
                        "name": "DummyAttack",
                        "params": {"fixed": {"level": 2}},
                    }
                ],
            },
        ],
        "metrics": [
            {
                "id": "metric",
                "name": "DummyMetric",
                "checkpoint": "after_attack",
                "inputs": {
                    "left": "embedded_image",
                    "right": "attacked_image",
                },
            }
        ],
    }


def dummy_catalog(*, include_bad_signature=False, include_unsupported=False):
    attacks = {"DummyAttack": DummyAttack}
    specs = [DUMMY_EMBEDDING_SPEC, DUMMY_ATTACK_SPEC, DUMMY_METRIC_SPEC]
    if include_bad_signature:
        attacks["BadSignatureAttack"] = BadSignatureAttack
        specs.append(BAD_SIGNATURE_SPEC)
    if include_unsupported:
        attacks["UnsupportedAttack"] = UnsupportedAttack
    return SolutionCatalog.from_registries(
        attacks=attacks,
        embeddings={"DummyEmbedding": DummyEmbedding},
        metrics={"DummyMetric": DummyMetric},
        specs=specs,
    )


def issue_codes(error):
    return {issue.code for issue in error.value.issues}


def test_example_yaml_resolves_against_ready_solutions():
    config = load_experiment_config(
        "examples/pipeline/dct_jpeg.yaml",
        resolve_paths=False,
    )
    catalog = SolutionCatalog.discover(plugins=config.plugins)

    resolved = catalog.resolve(config)

    assert [item.name for item in resolved.embeddings] == ["DCT"]
    assert [step.name for step in resolved.attack_scenarios[1].steps] == ["Jpeg"]
    assert [item.name for item in resolved.metrics] == ["PSNR", "PSNR", "BER"]
    assert len(resolved.catalog_fingerprint) == 64
    assert catalog.available(SolutionKind.EMBEDDING) == ("DCT",)
    assert "Jpeg" in catalog.available(SolutionKind.ATTACK)
    assert catalog.available(SolutionKind.METRIC) == ("BER", "PSNR")


def test_valid_explicit_catalog_resolves_without_executing_methods():
    config = ExperimentConfig.model_validate(minimal_config())
    resolved = dummy_catalog().resolve(config)

    assert resolved.embeddings[0].implementation is DummyEmbedding
    assert resolved.attack_scenarios[1].steps[0].implementation is DummyAttack
    assert resolved.metrics[0].implementation is DummyMetric


def test_resolved_experiment_is_pickle_safe():
    resolved = dummy_catalog().resolve(ExperimentConfig.model_validate(minimal_config()))

    restored = pickle.loads(pickle.dumps(resolved))

    assert restored.catalog_fingerprint == resolved.catalog_fingerprint
    assert restored.embeddings[0].implementation is DummyEmbedding
    assert restored.metrics[0].inputs == resolved.metrics[0].inputs


def test_catalog_fingerprint_is_independent_of_registry_and_metadata_order():
    first = SolutionCatalog.from_registries(
        attacks={"DummyAttack": DummyAttack},
        embeddings={"DummyEmbedding": DummyEmbedding},
        metrics={"DummyMetric": DummyMetric},
        specs=[DUMMY_EMBEDDING_SPEC, DUMMY_ATTACK_SPEC, DUMMY_METRIC_SPEC],
    )
    second = SolutionCatalog.from_registries(
        metrics={"DummyMetric": DummyMetric},
        embeddings={"DummyEmbedding": DummyEmbedding},
        attacks={"DummyAttack": DummyAttack},
        specs=[DUMMY_METRIC_SPEC, DUMMY_ATTACK_SPEC, DUMMY_EMBEDDING_SPEC],
    )

    assert first.fingerprint == second.fingerprint


def test_catalog_snapshots_input_registries():
    attacks = {"DummyAttack": DummyAttack}
    catalog = SolutionCatalog.from_registries(attacks=attacks, specs=[DUMMY_ATTACK_SPEC])

    attacks.clear()

    assert catalog.registered(SolutionKind.ATTACK) == ("DummyAttack",)


def test_unknown_solution_is_reported_with_configuration_path():
    data = minimal_config()
    data["embeddings"][0]["name"] = "MissingEmbedding"

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert error.value.issues[0].path == "embeddings[0].name"
    assert "unknown_solution" in issue_codes(error)


def test_solution_registered_in_wrong_category_is_distinguished_from_unknown():
    data = minimal_config()
    data["attack_scenarios"][1]["steps"][0]["name"] = "DummyMetric"

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert "wrong_solution_kind" in issue_codes(error)


def test_registered_solution_without_metadata_is_rejected():
    data = minimal_config()
    data["attack_scenarios"][1]["steps"][0]["name"] = "UnsupportedAttack"

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog(include_unsupported=True).resolve(ExperimentConfig.model_validate(data))

    assert "unsupported_solution" in issue_codes(error)


def test_missing_entry_point_is_reported():
    class MissingMethodAttack:
        pass

    spec = SolutionSpec(
        name="MissingMethodAttack",
        kind=SolutionKind.ATTACK,
        operations={"attack": SolutionOperationSpec(entry_point="attack")},
    )
    data = minimal_config()
    data["attack_scenarios"][1]["steps"][0]["name"] = "MissingMethodAttack"
    catalog = SolutionCatalog.from_registries(
        attacks={"MissingMethodAttack": MissingMethodAttack},
        embeddings={"DummyEmbedding": DummyEmbedding},
        metrics={"DummyMetric": DummyMetric},
        specs=[DUMMY_EMBEDDING_SPEC, spec, DUMMY_METRIC_SPEC],
    )

    with pytest.raises(SemanticValidationError) as error:
        catalog.resolve(ExperimentConfig.model_validate(data))

    assert "missing_entry_point" in issue_codes(error)


def test_entry_point_must_accept_kwargs_when_signature_is_available():
    data = minimal_config()
    data["attack_scenarios"][1]["steps"][0]["name"] = "BadSignatureAttack"

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog(include_bad_signature=True).resolve(ExperimentConfig.model_validate(data))

    assert "invalid_entry_point_signature" in issue_codes(error)


@pytest.mark.parametrize(
    ("parameter_declaration", "expected_path"),
    [
        ({"fixed": {"typo": 1}}, ".params.fixed.typo"),
        ({"grid": {"typo": [1, 2]}}, ".params.grid.typo[0]"),
        ({"variants": [{"typo": 1}, {"typo": 2}]}, ".params.variants[0].typo"),
    ],
)
def test_unknown_parameters_are_rejected_in_every_parameter_space(parameter_declaration, expected_path):
    data = minimal_config()
    data["attack_scenarios"][1]["steps"][0]["params"] = parameter_declaration

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert "unknown_parameter" in issue_codes(error)
    assert error.value.issues[0].path.endswith(expected_path)


def test_required_parameter_must_be_declared():
    data = minimal_config()
    del data["embeddings"][0]["embedding"]["params"]["fixed"]["mode"]

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert "missing_parameter" in issue_codes(error)


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_integer_parameter_validation_is_strict(value):
    data = minimal_config()
    data["attack_scenarios"][1]["steps"][0]["params"]["fixed"]["level"] = value

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert "invalid_parameter" in issue_codes(error)


@pytest.mark.parametrize("value", [-1, 11])
def test_numeric_parameter_bounds_are_checked(value):
    data = minimal_config()
    data["attack_scenarios"][1]["steps"][0]["params"]["fixed"]["level"] = value

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert "invalid_parameter" in issue_codes(error)


def test_boolean_parameter_validation_is_strict():
    data = minimal_config()
    data["attack_scenarios"][1]["steps"][0]["params"]["fixed"]["enabled"] = 1

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert "invalid_parameter" in issue_codes(error)


def test_linked_scalar_parameters_accept_numerically_equal_int_and_float():
    data = minimal_config()
    data["embeddings"][0]["embedding"]["params"]["fixed"]["strength"] = 1
    data["embeddings"][0]["extraction"]["params"]["fixed"]["strength"] = 1.0

    resolved = dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert resolved.embeddings[0].name == "DummyEmbedding"


def test_linked_parameters_must_have_equal_values():
    data = minimal_config()
    data["embeddings"][0]["extraction"]["params"]["fixed"]["strength"] = 2.0

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert "linked_parameter_mismatch" in issue_codes(error)


def test_linked_parameter_grids_preserve_axis_order():
    data = minimal_config()
    data["embeddings"][0]["embedding"]["params"] = {
        "fixed": {"mode": "robust"},
        "grid": {"strength": [1.0, 2.0]},
    }
    data["embeddings"][0]["extraction"]["params"] = {
        "grid": {"strength": [2.0, 1.0]}
    }

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert "linked_parameter_mismatch" in issue_codes(error)


def test_metric_checkpoint_is_checked_against_metadata():
    data = minimal_config()
    data["metrics"][0]["checkpoint"] = "after_extraction"

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert "unsupported_checkpoint" in issue_codes(error)


def test_unknown_metric_input_is_rejected():
    data = minimal_config()
    data["metrics"][0]["inputs"]["unexpected"] = "attacked_image"

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert "unknown_metric_input" in issue_codes(error)


def test_required_metric_input_must_be_bound():
    data = minimal_config()
    del data["metrics"][0]["inputs"]["right"]

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert "missing_metric_input" in issue_codes(error)


def test_metric_artifact_must_match_the_argument_contract():
    data = minimal_config()
    data["metrics"][0]["inputs"]["right"] = "original_image"

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert "incompatible_artifact" in issue_codes(error)


def test_all_independent_semantic_failures_are_aggregated():
    data = minimal_config()
    data["embeddings"][0]["name"] = "MissingEmbedding"
    data["attack_scenarios"][1]["steps"][0]["params"] = {"fixed": {"typo": 1}}
    data["metrics"][0]["inputs"]["right"] = "original_image"

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert issue_codes(error) == {
        "unknown_solution",
        "unknown_parameter",
        "incompatible_artifact",
    }
    assert "3 issues" in str(error.value)


def test_configuration_plugins_must_be_loaded_by_the_catalog():
    data = minimal_config()
    data["plugins"] = ["example.plugin"]

    with pytest.raises(SemanticValidationError) as error:
        dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert "plugin_not_loaded" in issue_codes(error)


def test_class_level_pipeline_spec_is_discovered_by_explicit_catalog():
    class ClassMetadataAttack:
        PIPELINE_SPEC = SolutionSpec(
            name="ClassMetadataAttack",
            kind=SolutionKind.ATTACK,
            operations={"attack": SolutionOperationSpec(entry_point="attack", runtime_arguments=("input_image",))},
        )

        @staticmethod
        def attack(**args):
            return args

    catalog = SolutionCatalog.from_registries(attacks={"ClassMetadataAttack": ClassMetadataAttack})

    assert catalog.available(SolutionKind.ATTACK) == ("ClassMetadataAttack",)


def test_class_metadata_name_must_match_registered_class():
    class ActualName:
        PIPELINE_SPEC = SolutionSpec(
            name="DifferentName",
            kind=SolutionKind.ATTACK,
            operations={"attack": SolutionOperationSpec(entry_point="attack")},
        )

        @staticmethod
        def attack(**args):
            return args

    with pytest.raises(SolutionDiscoveryError, match="registered class"):
        SolutionCatalog.from_registries(attacks={"ActualName": ActualName})


def test_conflicting_explicit_metadata_is_rejected():
    conflicting = SolutionSpec(
        name="DummyAttack",
        kind=SolutionKind.ATTACK,
        operations={
            "attack": SolutionOperationSpec(
                entry_point="attack",
                parameters={"another": ParameterSpec(ParameterKind.INTEGER, default=1)},
            )
        },
    )

    with pytest.raises(SolutionConflictError):
        SolutionCatalog.from_registries(
            attacks={"DummyAttack": DummyAttack},
            specs=[DUMMY_ATTACK_SPEC, conflicting],
        )


def test_solution_metadata_mappings_are_immutable():
    with pytest.raises(TypeError, match="immutable"):
        DUMMY_ATTACK_SPEC.operations["new"] = SolutionOperationSpec(entry_point="attack")

    with pytest.raises(TypeError, match="immutable"):
        DUMMY_ATTACK_SPEC.operations["attack"].parameters["new"] = ParameterSpec(
            ParameterKind.INTEGER,
            default=1,
        )


@pytest.fixture
def preserve_global_registries():
    snapshots = {
        Attack_Core: dict(Attack_Core.get_registered_attacks()),
        Embedding_Core: dict(Embedding_Core.get_registered_embeddings()),
        Expertise_Core: dict(Expertise_Core.get_registered_expertises()),
    }
    yield
    for core, snapshot in snapshots.items():
        getter = {
            Attack_Core: Attack_Core.get_registered_attacks,
            Embedding_Core: Embedding_Core.get_registered_embeddings,
            Expertise_Core: Expertise_Core.get_registered_expertises,
        }[core]
        getter().clear()
        getter().update(snapshot)


def write_plugin(tmp_path: Path, name: str, source: str) -> str:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return name


def test_plugin_class_metadata_is_imported(tmp_path, monkeypatch, preserve_global_registries):
    module_name = write_plugin(
        tmp_path,
        "dwarf_test_plugin_class_metadata",
        """
from dwarf.core.attack_orchestrator.attack_core import Ready_Geometric_Attacks
from dwarf.pipeline import SolutionKind, SolutionOperationSpec, SolutionSpec

class PluginAttack(Ready_Geometric_Attacks):
    PIPELINE_SPEC = SolutionSpec(
        name="PluginAttack",
        kind=SolutionKind.ATTACK,
        operations={"attack": SolutionOperationSpec(entry_point="attack", runtime_arguments=("input_image",))},
    )

    @staticmethod
    def attack(**args):
        return args["input_image"]
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    catalog = SolutionCatalog.discover(plugins=(module_name,))

    assert "PluginAttack" in catalog.available(SolutionKind.ATTACK)
    sys.modules.pop(module_name, None)


def test_plugin_module_metadata_is_imported(tmp_path, monkeypatch, preserve_global_registries):
    module_name = write_plugin(
        tmp_path,
        "dwarf_test_plugin_module_metadata",
        """
from dwarf.core.expertise_orchestrator.expertise_core import Ready_Robustness_Expertise
from dwarf.pipeline import SolutionKind, SolutionOperationSpec, SolutionSpec

class PluginMetric(Ready_Robustness_Expertise):
    @staticmethod
    def expertise(**args):
        return 0.0

DWARF_PIPELINE_SPECS = (
    SolutionSpec(
        name="PluginMetric",
        kind=SolutionKind.METRIC,
        operations={"expertise": SolutionOperationSpec(entry_point="expertise")},
    ),
)
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    catalog = SolutionCatalog.discover(plugins=(module_name,))

    assert "PluginMetric" in catalog.available(SolutionKind.METRIC)
    sys.modules.pop(module_name, None)


def test_plugin_cannot_replace_registered_solution(tmp_path, monkeypatch, preserve_global_registries):
    module_name = write_plugin(
        tmp_path,
        "dwarf_test_plugin_conflict",
        """
from dwarf.core.attack_orchestrator.attack_core import Ready_Compression_Attacks

class Jpeg(Ready_Compression_Attacks):
    @staticmethod
    def attack(**args):
        return args["input_image"]
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(SolutionConflictError, match="replaced registered solutions"):
        SolutionCatalog.discover(plugins=(module_name,))

    sys.modules.pop(module_name, None)


def test_plugin_import_failure_has_context(tmp_path, monkeypatch, preserve_global_registries):
    module_name = write_plugin(
        tmp_path,
        "dwarf_test_plugin_failure",
        'raise RuntimeError("plugin exploded")\n',
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(SolutionDiscoveryError, match="plugin exploded"):
        SolutionCatalog.discover(plugins=(module_name,))

    sys.modules.pop(module_name, None)


def test_config_can_be_copied_without_catalog_side_effects():
    data = minimal_config()
    original = copy.deepcopy(data)

    dummy_catalog().resolve(ExperimentConfig.model_validate(data))

    assert data == original
