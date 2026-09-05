"""Tests for deterministic, lazy experiment planning."""

import copy
import hashlib
import json
import pickle
from dataclasses import replace
from itertools import islice
from pathlib import Path

import pytest

from dwarf.pipeline import (
    ArtifactInputSpec,
    ArtifactKind,
    DataContract,
    DatasetManifest,
    ExperimentConfig,
    ExperimentPlan,
    ExperimentPlanner,
    ParameterKind,
    ParameterLinkSpec,
    ParameterSpec,
    PlanTooLargeError,
    PlanningError,
    SampleReference,
    SolutionCatalog,
    SolutionKind,
    SolutionOperationSpec,
    SolutionSpec,
    build_experiment_plan,
    derive_stable_seed,
    load_experiment_config,
    make_sample_id,
)


class PlannerEmbedding:
    @staticmethod
    def embedding(**args):
        raise AssertionError("planning must not execute embedding solutions")

    @staticmethod
    def extraction(**args):
        raise AssertionError("planning must not execute extraction solutions")


class PlannerEmbeddingTwo:
    @staticmethod
    def embedding(**args):
        raise AssertionError("planning must not execute embedding solutions")

    @staticmethod
    def extraction(**args):
        raise AssertionError("planning must not execute extraction solutions")


class PlannerAttack:
    @staticmethod
    def attack(**args):
        raise AssertionError("planning must not execute attack solutions")


class PlannerAttackTwo:
    @staticmethod
    def attack(**args):
        raise AssertionError("planning must not execute attack solutions")


class PlannerMetric:
    @staticmethod
    def expertise(**args):
        raise AssertionError("planning must not execute metric solutions")


PLANNER_EMBEDDING_SPEC = SolutionSpec(
    name="PlannerEmbedding",
    kind=SolutionKind.EMBEDDING,
    operations={
        "embedding": SolutionOperationSpec(
            entry_point="embedding",
            parameters={
                "margin": ParameterSpec(ParameterKind.NUMBER, default=150.0, minimum=0.0),
                "threshold": ParameterSpec(ParameterKind.NUMBER, default=25.0, minimum=0.0),
                "mode": ParameterSpec(
                    ParameterKind.STRING,
                    default="robust",
                    choices=("robust", "fast"),
                ),
            },
            runtime_arguments=("input_image", "watermark_bits"),
        ),
        "extraction": SolutionOperationSpec(
            entry_point="extraction",
            parameters={
                "threshold": ParameterSpec(ParameterKind.NUMBER, default=25.0, minimum=0.0),
                "decoder": ParameterSpec(
                    ParameterKind.STRING,
                    default="hard",
                    choices=("hard", "soft"),
                ),
            },
            runtime_arguments=("input_image", "num_bits"),
        ),
    },
    parameter_links=(
        ParameterLinkSpec(
            source_operation="embedding",
            source_parameter="threshold",
            target_operation="extraction",
            target_parameter="threshold",
        ),
    ),
    contracts={
        "input": DataContract.RGB_UINT8,
        "original_watermark": DataContract.BINARY_BITS,
        "extracted_watermark": DataContract.TERNARY_BITS,
    },
)

PLANNER_EMBEDDING_TWO_SPEC = SolutionSpec(
    name="PlannerEmbeddingTwo",
    kind=SolutionKind.EMBEDDING,
    operations={
        "embedding": SolutionOperationSpec(
            entry_point="embedding",
            parameters={
                "strength": ParameterSpec(ParameterKind.INTEGER, default=2, minimum=1),
            },
            runtime_arguments=("input_image", "watermark_bits"),
        ),
        "extraction": SolutionOperationSpec(
            entry_point="extraction",
            runtime_arguments=("input_image", "num_bits"),
        ),
    },
)

PLANNER_ATTACK_SPEC = SolutionSpec(
    name="PlannerAttack",
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

PLANNER_ATTACK_TWO_SPEC = SolutionSpec(
    name="PlannerAttackTwo",
    kind=SolutionKind.ATTACK,
    operations={
        "attack": SolutionOperationSpec(
            entry_point="attack",
            parameters={
                "sigma": ParameterSpec(ParameterKind.NUMBER, default=0.5, minimum=0.0),
                "channels": ParameterSpec(ParameterKind.STRING_SEQUENCE, default=("Y",)),
                "options": ParameterSpec(ParameterKind.JSON, default={"clip": True}),
            },
            runtime_arguments=("input_image",),
        )
    },
    deterministic=False,
)

_IMAGE_ARTIFACTS = ("original_image", "embedded_image", "attacked_image")
PLANNER_METRIC_SPEC = SolutionSpec(
    name="PlannerMetric",
    kind=SolutionKind.METRIC,
    operations={
        "expertise": SolutionOperationSpec(
            entry_point="expertise",
            parameters={
                "normalise": ParameterSpec(ParameterKind.BOOLEAN, default=False),
                "scale": ParameterSpec(ParameterKind.NUMBER, default=1.0, minimum=0.0),
            },
            artifact_inputs={
                "left": ArtifactInputSpec(_IMAGE_ARTIFACTS, ArtifactKind.IMAGE),
                "right": ArtifactInputSpec(_IMAGE_ARTIFACTS, ArtifactKind.IMAGE),
            },
            checkpoints=("after_embedding", "after_attack", "after_extraction"),
        )
    },
)


def planner_catalog():
    return SolutionCatalog.from_registries(
        attacks={
            "PlannerAttack": PlannerAttack,
            "PlannerAttackTwo": PlannerAttackTwo,
        },
        embeddings={
            "PlannerEmbedding": PlannerEmbedding,
            "PlannerEmbeddingTwo": PlannerEmbeddingTwo,
        },
        metrics={"PlannerMetric": PlannerMetric},
        specs=(
            PLANNER_EMBEDDING_SPEC,
            PLANNER_EMBEDDING_TWO_SPEC,
            PLANNER_ATTACK_SPEC,
            PLANNER_ATTACK_TWO_SPEC,
            PLANNER_METRIC_SPEC,
        ),
    )


def base_config_data(root: Path):
    return {
        "schema_version": 1,
        "experiment": {
            "name": "planner-test",
            "seed": 17,
            "repeats": 2,
            "max_cases": 10_000,
        },
        "dataset": {
            "type": "directory",
            "path": root,
            "recursive": True,
            "extensions": [".png"],
            "include": [],
            "exclude": [],
            "shuffle": False,
        },
        "watermark": {"type": "fixed_bits", "bits": "1010"},
        "embeddings": [
            {
                "id": "primary",
                "name": "PlannerEmbedding",
            }
        ],
        "attack_scenarios": [
            {"id": "clean", "steps": []},
            {
                "id": "chain",
                "description": "Two-step Cartesian attack chain.",
                "steps": [
                    {
                        "name": "PlannerAttack",
                        "params": {"grid": {"level": [1, 2]}},
                    },
                    {
                        "name": "PlannerAttackTwo",
                        "params": {"grid": {"sigma": [0.1, 0.2, 0.3]}},
                    },
                ],
            },
        ],
        "metrics": [
            {
                "id": "embedding-metric",
                "name": "PlannerMetric",
                "checkpoint": "after_embedding",
                "inputs": {"left": "original_image", "right": "embedded_image"},
            },
            {
                "id": "attack-metric",
                "name": "PlannerMetric",
                "checkpoint": "after_attack",
                "inputs": {"left": "embedded_image", "right": "attacked_image"},
                "params": {"grid": {"scale": [1.0, 2.0]}},
            },
            {
                "id": "extraction-metric",
                "name": "PlannerMetric",
                "checkpoint": "after_extraction",
                "inputs": {"left": "original_image", "right": "attacked_image"},
            },
        ],
    }


def make_manifest(config, names=("a.png", "nested/b.png"), *, checksum_prefix="sample"):
    root = config.dataset.path.expanduser().resolve(strict=False)
    samples = []
    for index, name in enumerate(names):
        checksum = hashlib.sha256(f"{checksum_prefix}:{name}".encode()).hexdigest()
        samples.append(
            SampleReference(
                sample_id=make_sample_id(name, checksum),
                relative_path=name,
                path=root.joinpath(*name.split("/")),
                size_bytes=index + 10,
                modified_time_ns=index + 100,
                checksum_sha256=checksum,
            )
        )
    return DatasetManifest(
        root=root,
        samples=tuple(samples),
        recursive=config.dataset.recursive,
        extensions=config.dataset.extensions,
        include=config.dataset.include,
        exclude=config.dataset.exclude,
        shuffled=config.dataset.shuffle,
        seed=config.experiment.seed,
        limit=config.dataset.limit,
    )


def make_plan(tmp_path, data=None, names=("a.png", "nested/b.png")):
    if data is None:
        data = base_config_data(tmp_path / "images")
    config = ExperimentConfig.model_validate(data)
    resolved = planner_catalog().resolve(config)
    manifest = make_manifest(config, names)
    return build_experiment_plan(resolved, manifest)


def test_example_yaml_produces_expected_counts_without_loading_images():
    config = load_experiment_config(
        "examples/pipeline/dct_jpeg.yaml",
        resolve_paths=False,
    )
    resolved = SolutionCatalog.discover(plugins=config.plugins).resolve(config)
    manifest = make_manifest(config, names=("example.png",))

    plan = build_experiment_plan(resolved, manifest)

    assert plan.counts.sample_count == 1
    assert plan.counts.embedding_variant_count == 3
    assert plan.counts.attack_variant_count == 5
    assert plan.counts.case_count == 15
    assert plan.counts.work_unit_count == 3
    assert plan.counts.embedding_execution_count == 3
    assert plan.counts.attack_execution_count == 12
    assert plan.counts.extraction_execution_count == 15
    assert plan.counts.metric_variant_counts == {
        "after_embedding": 1,
        "after_attack": 1,
        "after_extraction": 1,
    }
    assert plan.counts.metric_evaluation_count == 33


def test_planner_constructor_and_convenience_wrapper_are_equivalent(tmp_path):
    config = ExperimentConfig.model_validate(base_config_data(tmp_path / "images"))
    resolved = planner_catalog().resolve(config)
    manifest = make_manifest(config)

    direct = ExperimentPlanner(resolved, manifest).build()
    wrapped = build_experiment_plan(resolved, manifest)

    assert direct == wrapped


def test_plan_is_pickle_safe_and_has_compact_json(tmp_path):
    plan = make_plan(tmp_path)

    restored = pickle.loads(pickle.dumps(plan))
    payload = json.loads(plan.to_json())

    assert restored == plan
    assert restored.fingerprint == plan.fingerprint
    assert payload["fingerprint"] == plan.fingerprint
    assert payload["counts"]["case_count"] == 28
    assert "cases" not in payload
    assert "planner-test" in plan.summary()
    assert "Experiment cases: 28" in plan.summary()


def test_plan_rejects_inconsistent_identity_and_count_records(tmp_path):
    plan = make_plan(tmp_path)

    with pytest.raises(ValueError, match="fingerprint does not match"):
        replace(plan, fingerprint="0" * 64)
    with pytest.raises(ValueError, match="case_context_id does not match"):
        replace(plan, case_context_id="0" * 64)
    with pytest.raises(ValueError, match="sample_count"):
        replace(plan, counts=replace(plan.counts, sample_count=plan.counts.sample_count + 1))
    with pytest.raises(ValueError, match="metric evaluation counts"):
        changed_evaluations = dict(plan.counts.metric_evaluation_counts)
        changed_evaluations["after_attack"] += 1
        replace(
            plan,
            counts=replace(
                plan.counts,
                metric_evaluation_counts=changed_evaluations,
            ),
        )


def test_default_parameters_are_resolved_into_variants(tmp_path):
    plan = make_plan(tmp_path)

    embedding = next(plan.iter_embedding_variants())
    attacks = list(plan.iter_attack_variants())
    metrics = list(plan.iter_metric_variants())

    assert embedding.embedding_parameters == {
        "margin": 150.0,
        "mode": "robust",
        "threshold": 25.0,
    }
    assert embedding.extraction_parameters == {
        "decoder": "hard",
        "threshold": 25.0,
    }
    assert attacks[1].steps[0].parameters["enabled"] is True
    assert attacks[1].steps[1].parameters["channels"] == ("Y",)
    assert attacks[1].steps[1].parameters["options"] == {"clip": True}
    assert metrics[0].parameters == {"normalise": False, "scale": 1.0}


def test_fixed_parameters_override_metadata_defaults(tmp_path):
    data = base_config_data(tmp_path / "images")
    data["embeddings"][0]["embedding"] = {
        "params": {"fixed": {"margin": 90, "mode": "fast", "threshold": 8}}
    }
    data["embeddings"][0]["extraction"] = {
        "params": {"fixed": {"decoder": "soft", "threshold": 8}}
    }

    variant = next(make_plan(tmp_path, data).iter_embedding_variants())

    assert variant.embedding_parameters == {
        "margin": 90.0,
        "mode": "fast",
        "threshold": 8.0,
    }
    assert variant.extraction_parameters == {
        "decoder": "soft",
        "threshold": 8.0,
    }


def test_grid_is_expanded_as_deterministic_cartesian_product(tmp_path):
    data = base_config_data(tmp_path / "images")
    data["embeddings"][0]["embedding"] = {
        "params": {
            "grid": {
                "mode": ["robust", "fast"],
                "margin": [100, 200],
            }
        }
    }

    variants = list(make_plan(tmp_path, data).iter_embedding_variants())

    assert [
        (item.embedding_parameters["margin"], item.embedding_parameters["mode"])
        for item in variants
    ] == [
        (100.0, "robust"),
        (100.0, "fast"),
        (200.0, "robust"),
        (200.0, "fast"),
    ]
    assert [item.ordinal for item in variants] == list(range(4))


def test_explicit_variants_preserve_correlated_combinations(tmp_path):
    data = base_config_data(tmp_path / "images")
    data["embeddings"][0]["embedding"] = {
        "params": {
            "variants": [
                {"margin": 100, "mode": "robust"},
                {"margin": 220, "mode": "fast"},
            ]
        }
    }

    variants = list(make_plan(tmp_path, data).iter_embedding_variants())

    assert [
        (item.embedding_parameters["margin"], item.embedding_parameters["mode"])
        for item in variants
    ] == [(100.0, "robust"), (220.0, "fast")]


def test_linked_embedding_and_extraction_grids_are_zipped_not_multiplied(tmp_path):
    data = base_config_data(tmp_path / "images")
    data["embeddings"][0]["embedding"] = {
        "params": {
            "grid": {
                "margin": [100, 200, 300],
                "threshold": [10, 20],
            }
        }
    }
    data["embeddings"][0]["extraction"] = {
        "params": {"grid": {"threshold": [10, 20]}}
    }

    variants = list(make_plan(tmp_path, data).iter_embedding_variants())

    assert len(variants) == 6
    assert all(
        item.embedding_parameters["threshold"] == item.extraction_parameters["threshold"]
        for item in variants
    )
    assert [
        (item.embedding_parameters["margin"], item.embedding_parameters["threshold"])
        for item in variants
    ] == [
        (100.0, 10.0),
        (100.0, 20.0),
        (200.0, 10.0),
        (200.0, 20.0),
        (300.0, 10.0),
        (300.0, 20.0),
    ]


def test_linked_variant_dimensions_are_zipped(tmp_path):
    data = base_config_data(tmp_path / "images")
    data["embeddings"][0]["embedding"] = {
        "params": {
            "variants": [
                {"mode": "robust", "threshold": 10},
                {"mode": "fast", "threshold": 20},
            ]
        }
    }
    data["embeddings"][0]["extraction"] = {
        "params": {
            "variants": [
                {"decoder": "hard", "threshold": 10},
                {"decoder": "soft", "threshold": 20},
            ]
        }
    }

    variants = list(make_plan(tmp_path, data).iter_embedding_variants())

    assert len(variants) == 2
    assert [
        (
            item.embedding_parameters["mode"],
            item.embedding_parameters["threshold"],
            item.extraction_parameters["decoder"],
            item.extraction_parameters["threshold"],
        )
        for item in variants
    ] == [
        ("robust", 10.0, "hard", 10.0),
        ("fast", 20.0, "soft", 20.0),
    ]


def test_independent_embedding_and_extraction_axes_form_a_product(tmp_path):
    data = base_config_data(tmp_path / "images")
    data["embeddings"][0]["embedding"] = {
        "params": {"grid": {"margin": [100, 200]}}
    }
    data["embeddings"][0]["extraction"] = {
        "params": {"grid": {"decoder": ["hard", "soft"]}}
    }

    variants = list(make_plan(tmp_path, data).iter_embedding_variants())

    assert len(variants) == 4
    assert [
        (item.embedding_parameters["margin"], item.extraction_parameters["decoder"])
        for item in variants
    ] == [
        (100.0, "hard"),
        (100.0, "soft"),
        (200.0, "hard"),
        (200.0, "soft"),
    ]


def test_multiple_embedding_declarations_have_global_ordinals_and_unique_ids(tmp_path):
    data = base_config_data(tmp_path / "images")
    data["embeddings"].append(
        {
            "id": "secondary",
            "name": "PlannerEmbeddingTwo",
            "embedding": {"params": {"grid": {"strength": [2, 3]}}},
        }
    )

    variants = list(make_plan(tmp_path, data).iter_embedding_variants())

    assert [item.declaration_id for item in variants] == ["primary", "secondary", "secondary"]
    assert [item.ordinal for item in variants] == [0, 1, 2]
    assert len({item.variant_id for item in variants}) == 3
    assert variants[1].embedding_parameters["strength"] == 2
    assert variants[2].embedding_parameters["strength"] == 3


def test_clean_scenario_has_one_empty_variant(tmp_path):
    clean = next(make_plan(tmp_path).iter_attack_variants())

    assert clean.scenario_id == "clean"
    assert clean.is_clean
    assert clean.steps == ()
    assert clean.ordinal == 0


def test_attack_chain_is_expanded_as_step_cartesian_product(tmp_path):
    attacks = list(make_plan(tmp_path).iter_attack_variants())
    chain = attacks[1:]

    assert len(chain) == 6
    assert [
        (item.steps[0].parameters["level"], item.steps[1].parameters["sigma"])
        for item in chain
    ] == [
        (1, 0.1),
        (1, 0.2),
        (1, 0.3),
        (2, 0.1),
        (2, 0.2),
        (2, 0.3),
    ]
    assert all(tuple(step.step_index for step in item.steps) == (0, 1) for item in chain)
    assert [item.ordinal for item in attacks] == list(range(7))
    assert not chain[0].steps[1].deterministic


def test_metric_variants_are_expanded_without_multiplying_cases(tmp_path):
    plan = make_plan(tmp_path)
    metrics = list(plan.iter_metric_variants())

    assert len(metrics) == 4
    assert plan.counts.metric_variant_count == 4
    assert plan.counts.case_count == 28
    assert [metric.binding_id for metric in metrics] == [
        "embedding-metric",
        "attack-metric",
        "attack-metric",
        "extraction-metric",
    ]
    assert [metric.parameters["scale"] for metric in metrics] == [1.0, 1.0, 2.0, 1.0]


def test_metric_checkpoint_filter_preserves_global_ordinals(tmp_path):
    plan = make_plan(tmp_path)

    all_metrics = list(plan.iter_metric_variants())
    attack_metrics = list(plan.iter_metric_variants("after_attack"))

    assert [item.ordinal for item in all_metrics] == [0, 1, 2, 3]
    assert [item.ordinal for item in attack_metrics] == [1, 2]
    with pytest.raises(ValueError, match="unknown metric checkpoint"):
        list(plan.iter_metric_variants("before_embedding"))


def test_default_plan_counts_are_exact(tmp_path):
    counts = make_plan(tmp_path).counts

    assert counts.to_dict() == {
        "sample_count": 2,
        "embedding_variant_count": 1,
        "attack_variant_count": 7,
        "repeat_count": 2,
        "case_count": 28,
        "work_unit_count": 4,
        "embedding_execution_count": 4,
        "attack_execution_count": 48,
        "extraction_execution_count": 28,
        "metric_variant_count": 4,
        "metric_variant_counts": {
            "after_embedding": 1,
            "after_attack": 2,
            "after_extraction": 1,
        },
        "metric_evaluation_count": 88,
        "metric_evaluation_counts": {
            "after_embedding": 4,
            "after_attack": 56,
            "after_extraction": 28,
        },
    }


def test_no_metrics_produce_zero_metric_counts(tmp_path):
    data = base_config_data(tmp_path / "images")
    data["metrics"] = []

    counts = make_plan(tmp_path, data).counts

    assert counts.metric_variant_count == 0
    assert counts.metric_evaluation_count == 0
    assert set(counts.metric_variant_counts.values()) == {0}
    assert set(counts.metric_evaluation_counts.values()) == {0}


def test_fixed_and_per_image_watermarks_share_attack_branches_in_work_units(tmp_path):
    fixed_plan = make_plan(tmp_path)
    per_image_data = base_config_data(tmp_path / "images")
    per_image_data["watermark"] = {
        "type": "random_bits",
        "length": 16,
        "scope": "per_image",
    }
    per_image_plan = make_plan(tmp_path, per_image_data)

    for plan in (fixed_plan, per_image_plan):
        units = list(plan.iter_work_units())
        assert len(units) == 4
        assert all(len(unit.cases) == 7 for unit in units)
        assert plan.counts.work_unit_count == 4
        assert plan.counts.embedding_execution_count == 4


def test_per_case_watermark_creates_one_work_unit_per_case(tmp_path):
    data = base_config_data(tmp_path / "images")
    data["watermark"] = {
        "type": "random_bits",
        "length": 16,
        "scope": "per_case",
    }

    plan = make_plan(tmp_path, data)
    units = list(plan.iter_work_units())

    assert plan.counts.case_count == 28
    assert plan.counts.work_unit_count == 28
    assert plan.counts.embedding_execution_count == 28
    assert len(units) == 28
    assert all(len(unit.cases) == 1 for unit in units)
    assert len({unit.watermark.watermark_id for unit in units}) == 28


def test_plan_limit_rejects_large_case_space_before_enumeration(tmp_path):
    data = base_config_data(tmp_path / "images")
    data["experiment"]["max_cases"] = 27

    with pytest.raises(PlanTooLargeError) as error:
        make_plan(tmp_path, data)

    assert error.value.case_count == 28
    assert error.value.max_cases == 27
    assert "allow_large_plan=true" in str(error.value)


def test_plan_limit_accepts_exact_boundary_and_explicit_override(tmp_path):
    exact = base_config_data(tmp_path / "exact")
    exact["experiment"]["max_cases"] = 28
    assert make_plan(tmp_path, exact).counts.case_count == 28

    override = base_config_data(tmp_path / "override")
    override["experiment"]["max_cases"] = 1
    override["experiment"]["allow_large_plan"] = True
    assert make_plan(tmp_path, override).counts.case_count == 28


def test_same_inputs_produce_identical_variants_cases_seeds_and_fingerprint(tmp_path):
    first = make_plan(tmp_path)
    second = make_plan(tmp_path)

    assert first.fingerprint == second.fingerprint
    assert list(first.iter_embedding_variants()) == list(second.iter_embedding_variants())
    assert list(first.iter_attack_variants()) == list(second.iter_attack_variants())
    assert list(first.iter_metric_variants()) == list(second.iter_metric_variants())
    assert list(first.iter_cases()) == list(second.iter_cases())


def test_all_generated_identifiers_are_lowercase_sha256_digests(tmp_path):
    plan = make_plan(tmp_path)
    values = [plan.fingerprint, plan.case_context_id]
    values.extend(item.variant_id for item in plan.iter_embedding_variants())
    values.extend(item.variant_id for item in plan.iter_attack_variants())
    values.extend(item.variant_id for item in plan.iter_metric_variants())
    for case in plan.iter_cases():
        values.extend(
            [
                case.case_id,
                case.work_unit_id,
                case.watermark.watermark_id,
            ]
        )

    assert values
    assert all(len(value) == 64 for value in values)
    assert all(set(value) <= set("0123456789abcdef") for value in values)


def test_case_ordinals_and_seeds_are_deterministic_and_unique(tmp_path):
    cases = list(make_plan(tmp_path).iter_cases())

    assert [case.ordinal for case in cases] == list(range(28))
    assert len({case.case_id for case in cases}) == 28
    assert len({case.case_seed for case in cases}) == 28
    assert all(0 <= case.case_seed <= 2**63 - 1 for case in cases)


def test_manifest_order_changes_case_order_and_plan_fingerprint(tmp_path):
    data = base_config_data(tmp_path / "images")
    config = ExperimentConfig.model_validate(data)
    resolved = planner_catalog().resolve(config)
    first_manifest = make_manifest(config, names=("a.png", "b.png"))
    second_manifest = make_manifest(config, names=("b.png", "a.png"))

    first = build_experiment_plan(resolved, first_manifest)
    second = build_experiment_plan(resolved, second_manifest)

    assert first.fingerprint != second.fingerprint
    assert next(first.iter_cases()).sample.relative_path == "a.png"
    assert next(second.iter_cases()).sample.relative_path == "b.png"


def test_relocating_identical_manifest_preserves_plan_fingerprint(tmp_path):
    first_data = base_config_data(tmp_path / "first")
    second_data = base_config_data(tmp_path / "second")
    first_config = ExperimentConfig.model_validate(first_data)
    second_config = ExperimentConfig.model_validate(second_data)
    first_resolved = planner_catalog().resolve(first_config)
    second_resolved = planner_catalog().resolve(second_config)

    first = build_experiment_plan(first_resolved, make_manifest(first_config))
    second = build_experiment_plan(second_resolved, make_manifest(second_config))

    assert first.manifest.fingerprint == second.manifest.fingerprint
    assert first.fingerprint == second.fingerprint
    assert [case.case_id for case in first.iter_cases()] == [
        case.case_id for case in second.iter_cases()
    ]


def test_cosmetic_and_output_settings_do_not_change_computational_plan(tmp_path):
    first_data = base_config_data(tmp_path / "images")
    second_data = copy.deepcopy(first_data)
    second_data["experiment"].update(
        {
            "name": "renamed-experiment",
            "description": "Cosmetic text.",
            "tags": ["renamed"],
        }
    )
    second_data["execution"] = {"backend": "serial", "show_progress": False}
    second_data["output"] = {
        "directory": tmp_path / "elsewhere",
        "save_images": "none",
        "exports": [],
    }
    second_data["report"] = {"enabled": False, "formats": []}

    first = make_plan(tmp_path, first_data)
    second = make_plan(tmp_path, second_data)

    assert first.fingerprint == second.fingerprint
    assert [case.case_id for case in first.iter_cases()] == [
        case.case_id for case in second.iter_cases()
    ]


def test_metric_changes_affect_plan_fingerprint_but_not_case_ids(tmp_path):
    first_data = base_config_data(tmp_path / "images")
    second_data = copy.deepcopy(first_data)
    second_data["metrics"][1]["params"] = {"fixed": {"scale": 3.0}}

    first = make_plan(tmp_path, first_data)
    second = make_plan(tmp_path, second_data)

    assert first.fingerprint != second.fingerprint
    assert [case.case_id for case in first.iter_cases()] == [
        case.case_id for case in second.iter_cases()
    ]


def test_attack_parameter_change_changes_attack_and_case_ids(tmp_path):
    first_data = base_config_data(tmp_path / "images")
    second_data = copy.deepcopy(first_data)
    second_data["attack_scenarios"][1]["steps"][0]["params"]["grid"]["level"] = [1, 3]

    first = make_plan(tmp_path, first_data)
    second = make_plan(tmp_path, second_data)

    assert [item.variant_id for item in first.iter_attack_variants()] != [
        item.variant_id for item in second.iter_attack_variants()
    ]
    assert [case.case_id for case in first.iter_cases()] != [
        case.case_id for case in second.iter_cases()
    ]


def test_preprocessing_change_changes_case_context_and_case_ids(tmp_path):
    first_data = base_config_data(tmp_path / "images")
    second_data = copy.deepcopy(first_data)
    second_data["dataset"]["preprocessing"] = {"resize": [64, 64]}

    first = make_plan(tmp_path, first_data)
    second = make_plan(tmp_path, second_data)

    assert first.case_context_id != second.case_context_id
    assert [case.case_id for case in first.iter_cases()] != [
        case.case_id for case in second.iter_cases()
    ]


def test_increasing_repeat_count_preserves_existing_case_ids(tmp_path):
    first_data = base_config_data(tmp_path / "images")
    second_data = copy.deepcopy(first_data)
    first_data["experiment"]["repeats"] = 1
    second_data["experiment"]["repeats"] = 3

    first = make_plan(tmp_path, first_data)
    second = make_plan(tmp_path, second_data)
    first_ids = {case.case_id for case in first.iter_cases()}
    second_ids_for_existing_repeat = {
        case.case_id for case in second.iter_cases() if case.repeat_index == 0
    }

    assert first.fingerprint != second.fingerprint
    assert first_ids == second_ids_for_existing_repeat


def test_fixed_watermark_reference_is_shared_and_contains_no_seed(tmp_path):
    cases = list(make_plan(tmp_path).iter_cases())

    assert len({case.watermark.watermark_id for case in cases}) == 1
    assert all(case.watermark.kind == "fixed_bits" for case in cases)
    assert all(case.watermark.scope == "fixed_for_run" for case in cases)
    assert all(case.watermark.length == 4 for case in cases)
    assert all(case.watermark.seed is None for case in cases)


def test_random_fixed_for_run_watermark_is_shared_and_seeded(tmp_path):
    data = base_config_data(tmp_path / "images")
    data["watermark"] = {
        "type": "random_bits",
        "length": 32,
        "scope": "fixed_for_run",
    }

    cases = list(make_plan(tmp_path, data).iter_cases())

    assert len({case.watermark.watermark_id for case in cases}) == 1
    assert len({case.watermark.seed for case in cases}) == 1
    assert all(case.watermark.length == 32 for case in cases)


def test_per_image_watermark_depends_only_on_sample(tmp_path):
    data = base_config_data(tmp_path / "images")
    data["watermark"] = {
        "type": "random_bits",
        "length": 32,
        "scope": "per_image",
    }
    data["embeddings"].append(
        {"id": "secondary", "name": "PlannerEmbeddingTwo"}
    )

    cases = list(make_plan(tmp_path, data).iter_cases())
    by_sample = {}
    for case in cases:
        by_sample.setdefault(case.sample.sample_id, set()).add(case.watermark.watermark_id)

    assert len(by_sample) == 2
    assert all(len(ids) == 1 for ids in by_sample.values())
    assert len({next(iter(ids)) for ids in by_sample.values()}) == 2


def test_global_seed_changes_random_watermark_and_case_ids(tmp_path):
    first_data = base_config_data(tmp_path / "images")
    first_data["watermark"] = {
        "type": "random_bits",
        "length": 32,
        "scope": "fixed_for_run",
    }
    second_data = copy.deepcopy(first_data)
    second_data["experiment"]["seed"] = 18

    first = make_plan(tmp_path, first_data)
    second = make_plan(tmp_path, second_data)

    assert next(first.iter_cases()).watermark != next(second.iter_cases()).watermark
    assert [case.case_id for case in first.iter_cases()] != [
        case.case_id for case in second.iter_cases()
    ]


def test_flattened_work_units_match_direct_case_iterator(tmp_path):
    for scope in ("fixed_for_run", "per_image", "per_case"):
        data = base_config_data(tmp_path / scope)
        data["watermark"] = {
            "type": "random_bits",
            "length": 8,
            "scope": scope,
        }
        plan = make_plan(tmp_path, data)

        direct = list(plan.iter_cases())
        flattened = [case for unit in plan.iter_work_units() for case in unit.cases]

        assert direct == flattened


def test_shared_work_unit_groups_same_sample_embedding_repeat_and_watermark(tmp_path):
    units = list(make_plan(tmp_path).iter_work_units())

    assert [unit.ordinal for unit in units] == list(range(4))
    for unit in units:
        assert len(unit.cases) == 7
        assert {case.sample for case in unit.cases} == {unit.sample}
        assert all(case.embedding == unit.embedding for case in unit.cases)
        assert {case.repeat_index for case in unit.cases} == {unit.repeat_index}
        assert {case.embedding_seed for case in unit.cases} == {unit.embedding_seed}
        assert {case.watermark for case in unit.cases} == {unit.watermark}
        assert {case.work_unit_id for case in unit.cases} == {unit.work_unit_id}


def test_embedding_seed_is_shared_within_a_work_unit_and_domain_separated_from_cases(tmp_path):
    plan = make_plan(tmp_path)

    for unit in plan.iter_work_units():
        assert {case.embedding_seed for case in unit.cases} == {unit.embedding_seed}
        assert all(case.embedding_seed != case.case_seed for case in unit.cases)

    repeated = make_plan(tmp_path)
    assert [case.embedding_seed for case in plan.iter_cases()] == [
        case.embedding_seed for case in repeated.iter_cases()
    ]


def test_per_case_watermarks_produce_independent_embedding_seeds(tmp_path):
    data = base_config_data(tmp_path / "images")
    data["watermark"] = {
        "type": "random_bits",
        "length": 8,
        "scope": "per_case",
    }
    plan = make_plan(tmp_path, data, names=("a.png",))

    cases = list(plan.iter_cases())

    assert len({case.embedding_seed for case in cases}) == len(cases)
    assert all(unit.embedding_seed == unit.cases[0].embedding_seed for unit in plan.iter_work_units())


def test_direct_case_iterator_does_not_delegate_to_work_unit_materialisation(tmp_path, monkeypatch):
    plan = make_plan(tmp_path)

    def fail_if_called(self):
        raise AssertionError("iter_cases must not materialise work units")

    monkeypatch.setattr(ExperimentPlan, "iter_work_units", fail_if_called)

    first_two = list(islice(plan.iter_cases(), 2))
    assert len(first_two) == 2


def test_build_and_iteration_never_execute_registered_solutions(tmp_path):
    plan = make_plan(tmp_path)

    list(plan.iter_embedding_variants())
    list(plan.iter_attack_variants())
    list(plan.iter_metric_variants())
    list(plan.iter_cases())


def test_derive_stable_seed_is_portable_and_domain_separated():
    first = derive_stable_seed("case", 42, "sample", {"b": 2, "a": 1})
    reordered = derive_stable_seed("case", 42, "sample", {"a": 1, "b": 2})
    changed_domain = derive_stable_seed("watermark", 42, "sample", {"a": 1, "b": 2})
    changed_part = derive_stable_seed("case", 43, "sample", {"a": 1, "b": 2})

    assert first == reordered
    assert first != changed_domain
    assert first != changed_part
    assert 0 <= first <= 2**63 - 1
    with pytest.raises(ValueError, match="domain must be a non-empty string"):
        derive_stable_seed("", 1)


@pytest.mark.parametrize(
    ("field", "replacement_value", "expected_fragment"),
    [
        ("root", "other-root", "root is"),
        ("recursive", False, "recursive is"),
        ("extensions", (".jpg",), "extensions are"),
        ("include", ("selected/**",), "include patterns are"),
        ("exclude", ("ignored/**",), "exclude patterns are"),
        ("shuffled", True, "shuffled is"),
        ("seed", 999, "seed is"),
        ("limit", 1, "limit is"),
    ],
)
def test_manifest_selection_must_match_configuration(
    tmp_path,
    field,
    replacement_value,
    expected_fragment,
):
    data = base_config_data(tmp_path / "images")
    config = ExperimentConfig.model_validate(data)
    resolved = planner_catalog().resolve(config)
    manifest = make_manifest(config)
    if field == "root":
        other_root = (tmp_path / replacement_value).resolve()
        samples = tuple(
            replace(
                sample,
                path=other_root.joinpath(*sample.relative_path.split("/")),
            )
            for sample in manifest.samples
        )
        manifest = replace(manifest, root=other_root, samples=samples)
    else:
        manifest = replace(manifest, **{field: replacement_value})

    with pytest.raises(PlanningError, match=expected_fragment):
        build_experiment_plan(resolved, manifest)


def test_resolved_embedding_must_match_original_configuration(tmp_path):
    data = base_config_data(tmp_path / "images")
    config = ExperimentConfig.model_validate(data)
    resolved = planner_catalog().resolve(config)
    manifest = make_manifest(config)
    changed_operation = resolved.embeddings[0].embedding.model_copy(
        update={"params": resolved.embeddings[0].embedding.params.model_copy(update={"fixed": {"margin": 50}})}
    )
    malformed = replace(
        resolved,
        embeddings=(replace(resolved.embeddings[0], embedding=changed_operation),),
    )

    with pytest.raises(PlanningError, match="embedding does not match"):
        build_experiment_plan(malformed, manifest)


def test_resolved_attack_and_metric_must_match_original_configuration(tmp_path):
    data = base_config_data(tmp_path / "images")
    config = ExperimentConfig.model_validate(data)
    resolved = planner_catalog().resolve(config)
    manifest = make_manifest(config)

    bad_step = replace(resolved.attack_scenarios[1].steps[0], index=7)
    bad_scenario = replace(
        resolved.attack_scenarios[1],
        steps=(bad_step, resolved.attack_scenarios[1].steps[1]),
    )
    bad_attacks = replace(
        resolved,
        attack_scenarios=(resolved.attack_scenarios[0], bad_scenario),
    )
    with pytest.raises(PlanningError, match=r"steps\[0\] does not match"):
        build_experiment_plan(bad_attacks, manifest)

    bad_metric = replace(resolved.metrics[0], inputs={"left": "embedded_image", "right": "embedded_image"})
    bad_metrics = replace(
        resolved,
        metrics=(bad_metric, *resolved.metrics[1:]),
    )
    with pytest.raises(PlanningError, match="inputs does not match"):
        build_experiment_plan(bad_metrics, manifest)


def test_resolved_plugins_and_solution_metadata_are_checked(tmp_path):
    data = base_config_data(tmp_path / "images")
    config = ExperimentConfig.model_validate(data)
    resolved = planner_catalog().resolve(config)
    manifest = make_manifest(config)

    with pytest.raises(PlanningError, match="resolved plugins"):
        build_experiment_plan(replace(resolved, plugins=("unexpected.plugin",)), manifest)

    wrong_implementation = replace(
        resolved,
        embeddings=(replace(resolved.embeddings[0], implementation=PlannerEmbeddingTwo),),
    )
    with pytest.raises(PlanningError, match="implementation is"):
        build_experiment_plan(wrong_implementation, manifest)

    wrong_spec = replace(PLANNER_EMBEDDING_SPEC, name="DifferentEmbedding")
    malformed_embedding = replace(resolved.embeddings[0], spec=wrong_spec)
    malformed = replace(resolved, embeddings=(malformed_embedding,))
    with pytest.raises(PlanningError, match="describes solution"):
        build_experiment_plan(malformed, manifest)


def test_planner_defensively_rejects_misaligned_link_axes(tmp_path):
    data = base_config_data(tmp_path / "images")
    data["embeddings"][0]["embedding"] = {
        "params": {"grid": {"threshold": [10, 20]}}
    }
    data["embeddings"][0]["extraction"] = {
        "params": {"grid": {"threshold": [10, 20]}}
    }
    config = ExperimentConfig.model_validate(data)
    resolved = planner_catalog().resolve(config)
    manifest = make_manifest(config)

    malformed_extraction = resolved.embeddings[0].extraction.model_copy(
        update={
            "params": resolved.embeddings[0].extraction.params.model_copy(
                update={"grid": {"threshold": (20, 10)}}
            )
        }
    )
    malformed = replace(
        resolved,
        embeddings=(replace(resolved.embeddings[0], extraction=malformed_extraction),),
        config=config.model_copy(
            update={
                "embeddings": (
                    config.embeddings[0].model_copy(update={"extraction": malformed_extraction}),
                )
            }
        ),
    )

    with pytest.raises(PlanningError, match="different ordered values"):
        build_experiment_plan(malformed, manifest)


def test_plan_too_large_error_validates_constructor_arguments():
    with pytest.raises(ValueError, match="case_count"):
        PlanTooLargeError(case_count=True, max_cases=1)
    with pytest.raises(ValueError, match="max_cases"):
        PlanTooLargeError(case_count=1, max_cases=-1)
