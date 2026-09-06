"""Tests for immutable serial-execution result records."""

import pickle

import pytest

from dwarf.pipeline.results import (
    CaseResult,
    ExecutionResult,
    MetricResult,
    OperationError,
    OperationTiming,
    ResultStatus,
    WorkUnitResult,
)


def successful_metric(case_id="case"):
    return MetricResult(
        metric_id="psnr",
        metric_variant_id="metric-variant",
        solution_name="PSNR",
        checkpoint="after_attack",
        work_unit_id="work-unit",
        case_id=case_id,
        status=ResultStatus.SUCCESS,
        value=42.0,
    )


def test_execution_result_aggregates_cases_and_metrics():
    unit_metric = successful_metric(case_id=None)
    case_metric = successful_metric()
    case = CaseResult(
        case_id="case",
        work_unit_id="work-unit",
        ordinal=0,
        sample_id="sample",
        embedding_variant_id="embedding",
        attack_variant_id="attack",
        repeat_index=0,
        status=ResultStatus.SUCCESS,
        metrics=(case_metric,),
    )
    unit = WorkUnitResult(
        work_unit_id="work-unit",
        sample_id="sample",
        embedding_variant_id="embedding",
        status=ResultStatus.SUCCESS,
        cases=(case,),
        metrics=(unit_metric,),
    )
    result = ExecutionResult(
        plan_fingerprint="plan",
        catalog_fingerprint="catalog",
        manifest_fingerprint="manifest",
        erasure_policy="count_as_error",
        work_units=(unit,),
        duration_seconds=0.1,
        started_at_utc="2026-01-01T00:00:00+00:00",
        finished_at_utc="2026-01-01T00:00:01+00:00",
    )

    assert result.cases == (case,)
    assert result.metrics == (unit_metric, case_metric)
    assert result.successful_case_count == 1
    assert result.successful_metric_count == 2
    assert result.to_dict()["summary"]["cases"] == 1
    assert pickle.loads(pickle.dumps(result)) == result


def test_non_successful_metric_requires_error():
    with pytest.raises(ValueError, match="requires an error"):
        MetricResult(
            metric_id="metric",
            metric_variant_id="variant",
            solution_name="Metric",
            checkpoint="after_attack",
            work_unit_id="unit",
            case_id="case",
            status=ResultStatus.FAILED,
            value=None,
        )


def test_invalid_timing_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        OperationTiming("stage", -1.0)


def test_operation_error_preserves_exception_type():
    error = OperationError.from_exception("embedding", ValueError("bad value"))

    assert error.stage == "embedding"
    assert error.exception_type.endswith("ValueError")
    assert error.message == "bad value"
