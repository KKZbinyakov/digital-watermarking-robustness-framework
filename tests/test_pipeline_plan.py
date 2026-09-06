"""Unit tests for immutable records emitted by the experiment planner."""

import hashlib
import json
import pickle
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from dwarf.pipeline import (
    AttackScenarioVariant,
    AttackStepVariant,
    EmbeddingVariant,
    ExperimentCase,
    MetricVariant,
    PlanCounts,
    SampleReference,
    WatermarkReference,
    WorkUnit,
    make_sample_id,
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sample_reference(tmp_path):
    checksum = digest("sample-bytes")
    root = (tmp_path / "images").resolve()
    return SampleReference(
        sample_id=make_sample_id("sample.png", checksum),
        relative_path="sample.png",
        path=root / "sample.png",
        size_bytes=20,
        modified_time_ns=30,
        checksum_sha256=checksum,
    )


def embedding_variant():
    return EmbeddingVariant(
        variant_id=digest("embedding"),
        declaration_id="dct",
        solution_name="DCT",
        ordinal=0,
        embedding_parameters={"margin": 150.0, "nested": {"shape": [8, 8]}},
        extraction_parameters={"threshold": 25.0},
        deterministic=True,
        resource="cpu",
    )


def attack_step(index=0):
    return AttackStepVariant(
        variant_id=digest(f"attack-step-{index}"),
        step_index=index,
        solution_name="Jpeg",
        parameter_ordinal=0,
        parameters={"quality": 75},
        deterministic=True,
        resource="cpu",
    )


def attack_variant():
    return AttackScenarioVariant(
        variant_id=digest("attack-scenario"),
        scenario_id="jpeg",
        description="JPEG compression.",
        ordinal=0,
        steps=(attack_step(),),
    )


def watermark_reference():
    return WatermarkReference(
        watermark_id=digest("watermark"),
        kind="random_bits",
        scope="fixed_for_run",
        length=64,
        seed=42,
    )


def experiment_case(tmp_path, **updates):
    data = {
        "case_id": digest("case"),
        "work_unit_id": digest("work-unit"),
        "ordinal": 0,
        "sample": sample_reference(tmp_path),
        "embedding": embedding_variant(),
        "attack": attack_variant(),
        "repeat_index": 0,
        "embedding_seed": 5,
        "case_seed": 7,
        "watermark": watermark_reference(),
    }
    data.update(updates)
    return ExperimentCase(**data)


def test_fixed_watermark_reference_contract():
    reference = WatermarkReference(
        watermark_id=digest("fixed"),
        kind="fixed_bits",
        scope="fixed_for_run",
        length=4,
    )

    assert reference.seed is None
    assert reference.to_dict()["kind"] == "fixed_bits"


@pytest.mark.parametrize(
    "updates",
    [
        {"watermark_id": "invalid"},
        {"kind": "bytes"},
        {"scope": "per_run"},
        {"length": 0},
        {"seed": -1},
        {"seed": 2**63},
    ],
)
def test_random_watermark_reference_rejects_invalid_values(updates):
    data = {
        "watermark_id": digest("random"),
        "kind": "random_bits",
        "scope": "per_case",
        "length": 8,
        "seed": 1,
    }
    data.update(updates)

    with pytest.raises((TypeError, ValueError)):
        WatermarkReference(**data)


def test_watermark_kind_specific_rules_are_enforced():
    with pytest.raises(ValueError, match="must not declare a seed"):
        WatermarkReference(
            watermark_id=digest("fixed"),
            kind="fixed_bits",
            scope="fixed_for_run",
            length=4,
            seed=1,
        )
    with pytest.raises(ValueError, match="require a deterministic seed"):
        WatermarkReference(
            watermark_id=digest("random"),
            kind="random_bits",
            scope="fixed_for_run",
            length=4,
        )
    with pytest.raises(ValueError, match="fixed_for_run"):
        WatermarkReference(
            watermark_id=digest("fixed-image"),
            kind="fixed_bits",
            scope="per_image",
            length=4,
        )


def test_embedding_variant_deeply_freezes_parameter_mappings():
    source = {"margin": 100.0, "nested": {"shape": [8, 8]}}
    variant = EmbeddingVariant(
        variant_id=digest("embedding-freeze"),
        declaration_id="dct",
        solution_name="DCT",
        ordinal=2,
        embedding_parameters=source,
        extraction_parameters={"threshold": 25.0},
        deterministic=True,
        resource="gpu",
    )
    source["margin"] = 999.0
    source["nested"]["shape"].append(16)

    assert variant.embedding_parameters["margin"] == 100.0
    assert variant.embedding_parameters["nested"]["shape"] == (8, 8)
    with pytest.raises(TypeError):
        variant.embedding_parameters["margin"] = 50.0
    assert variant.to_dict()["embedding_parameters"]["nested"]["shape"] == [8, 8]


@pytest.mark.parametrize(
    "updates",
    [
        {"variant_id": "bad"},
        {"declaration_id": ""},
        {"solution_name": ""},
        {"ordinal": -1},
        {"deterministic": 1},
        {"resource": "tpu"},
        {"embedding_parameters": {"value": float("nan")}},
        {"embedding_parameters": {1: "bad-key"}},
    ],
)
def test_embedding_variant_rejects_invalid_fields(updates):
    data = {
        "variant_id": digest("embedding-invalid"),
        "declaration_id": "dct",
        "solution_name": "DCT",
        "ordinal": 0,
        "embedding_parameters": {},
        "extraction_parameters": {},
        "deterministic": True,
        "resource": "cpu",
    }
    data.update(updates)

    with pytest.raises((TypeError, ValueError)):
        EmbeddingVariant(**data)


def test_attack_step_variant_normalises_parameters_and_metadata():
    step = AttackStepVariant(
        variant_id=digest("step-normalise"),
        step_index=1,
        solution_name="Noise",
        parameter_ordinal=3,
        parameters={"sigma": 0.2, "options": {"clip": True}},
        deterministic=False,
        resource="io",
    )

    assert step.step_index == 1
    assert step.parameter_ordinal == 3
    assert step.parameters["options"] == {"clip": True}
    assert step.to_dict()["parameters"]["options"] == {"clip": True}


def test_attack_scenario_requires_contiguous_step_indexes():
    with pytest.raises(ValueError, match="contiguous"):
        AttackScenarioVariant(
            variant_id=digest("bad-scenario"),
            scenario_id="chain",
            description=None,
            ordinal=0,
            steps=(attack_step(index=1),),
        )


def test_clean_attack_scenario_is_valid():
    clean = AttackScenarioVariant(
        variant_id=digest("clean"),
        scenario_id="clean",
        description=None,
        ordinal=0,
        steps=(),
    )

    assert clean.is_clean
    assert clean.to_dict()["steps"] == []


def test_metric_variant_validates_inputs_checkpoint_and_mappings():
    metric = MetricVariant(
        variant_id=digest("metric"),
        binding_id="psnr",
        solution_name="PSNR",
        checkpoint="after_attack",
        ordinal=0,
        inputs={"original_image": "embedded_image", "distorted_image": "attacked_image"},
        parameters={"data_range": 255},
        deterministic=True,
        resource="cpu",
    )

    assert metric.inputs["distorted_image"] == "attacked_image"
    assert metric.to_dict()["parameters"] == {"data_range": 255}
    with pytest.raises(TypeError):
        metric.inputs["new"] = "original_image"

    with pytest.raises(ValueError, match="unknown metric checkpoint"):
        replace(metric, checkpoint="before_attack")
    with pytest.raises(TypeError, match="strings to strings"):
        replace(metric, inputs={"image": 1})


def test_experiment_case_is_pickle_safe_and_json_compatible(tmp_path):
    case = experiment_case(tmp_path)

    restored = pickle.loads(pickle.dumps(case))
    payload = case.to_dict(include_absolute_path=False)

    assert restored == case
    assert payload["sample"]["relative_path"] == "sample.png"
    assert "path" not in payload["sample"]
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "bad"),
        ("work_unit_id", "bad"),
        ("ordinal", -1),
        ("sample", object()),
        ("embedding", object()),
        ("attack", object()),
        ("repeat_index", -1),
        ("embedding_seed", 2**63),
        ("case_seed", 2**63),
        ("watermark", object()),
    ],
)
def test_experiment_case_rejects_invalid_fields(tmp_path, field, value):
    with pytest.raises((TypeError, ValueError)):
        experiment_case(tmp_path, **{field: value})


def test_work_unit_requires_consistent_nonempty_cases(tmp_path):
    case = experiment_case(tmp_path)
    unit = WorkUnit(
        work_unit_id=case.work_unit_id,
        ordinal=0,
        sample=case.sample,
        embedding=case.embedding,
        repeat_index=case.repeat_index,
        embedding_seed=case.embedding_seed,
        watermark=case.watermark,
        cases=(case,),
    )

    assert unit.cases == (case,)
    assert unit.to_dict(include_absolute_path=False)["cases"][0]["case_id"] == case.case_id

    with pytest.raises(ValueError, match="at least one"):
        replace(unit, cases=())
    with pytest.raises(ValueError, match="duplicate case IDs"):
        replace(unit, cases=(case, case))
    with pytest.raises(ValueError, match="containing work unit"):
        replace(unit, work_unit_id=digest("other-unit"))


def test_work_unit_detects_case_context_mismatches(tmp_path):
    case = experiment_case(tmp_path)
    unit_data = {
        "work_unit_id": case.work_unit_id,
        "ordinal": 0,
        "sample": case.sample,
        "embedding": case.embedding,
        "repeat_index": case.repeat_index,
        "embedding_seed": case.embedding_seed,
        "watermark": case.watermark,
    }

    mismatched_cases = [
        replace(case, sample=sample_reference(tmp_path / "other")),
        replace(case, embedding=replace(case.embedding, variant_id=digest("other-embedding"))),
        replace(case, repeat_index=1),
        replace(case, embedding_seed=case.embedding_seed + 1),
        replace(case, watermark=replace(case.watermark, watermark_id=digest("other-watermark"))),
    ]
    for mismatched in mismatched_cases:
        with pytest.raises(ValueError):
            WorkUnit(cases=(mismatched,), **unit_data)


def test_plan_counts_calculates_totals_and_serialises():
    counts = PlanCounts(
        sample_count=2,
        embedding_variant_count=3,
        attack_variant_count=4,
        repeat_count=1,
        case_count=24,
        work_unit_count=6,
        embedding_execution_count=6,
        attack_execution_count=18,
        extraction_execution_count=24,
        metric_variant_counts={
            "after_embedding": 1,
            "after_attack": 2,
            "after_extraction": 3,
        },
        metric_evaluation_counts={
            "after_embedding": 6,
            "after_attack": 48,
            "after_extraction": 72,
        },
    )

    assert counts.metric_variant_count == 6
    assert counts.metric_evaluation_count == 126
    payload = json.loads(counts.to_json())
    assert payload["metric_variant_count"] == 6
    assert payload["metric_evaluation_count"] == 126
    with pytest.raises(TypeError):
        counts.metric_variant_counts["after_attack"] = 3


@pytest.mark.parametrize(
    "updates",
    [
        {"sample_count": -1},
        {"embedding_variant_count": 0},
        {"attack_variant_count": 0},
        {"repeat_count": 0},
        {"case_count": True},
        {"metric_variant_counts": {"after_embedding": 1}},
        {
            "metric_evaluation_counts": {
                "after_embedding": 1,
                "after_attack": 2,
                "after_extraction": -1,
            }
        },
    ],
)
def test_plan_counts_rejects_invalid_values(updates):
    data = {
        "sample_count": 1,
        "embedding_variant_count": 1,
        "attack_variant_count": 1,
        "repeat_count": 1,
        "case_count": 1,
        "work_unit_count": 1,
        "embedding_execution_count": 1,
        "attack_execution_count": 0,
        "extraction_execution_count": 1,
        "metric_variant_counts": {
            "after_embedding": 0,
            "after_attack": 0,
            "after_extraction": 0,
        },
        "metric_evaluation_counts": {
            "after_embedding": 0,
            "after_attack": 0,
            "after_extraction": 0,
        },
    }
    data.update(updates)

    with pytest.raises((TypeError, ValueError)):
        PlanCounts(**data)


def test_plan_records_are_frozen(tmp_path):
    objects = [
        watermark_reference(),
        embedding_variant(),
        attack_step(),
        attack_variant(),
        experiment_case(tmp_path),
    ]

    for value in objects:
        with pytest.raises(FrozenInstanceError):
            value.ordinal = 100
