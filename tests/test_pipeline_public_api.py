"""Regression tests for the public DWARF pipeline API."""

import dwarf.pipeline as pipeline_api
from dwarf.pipeline import Pipeline
from dwarf.pipeline.executors import (
    ExecutionAbortedError,
    ExecutionError,
    SerialExecutor,
    UnsupportedExecutionBackendError,
)
from dwarf.pipeline.pipeline import Pipeline as PipelineImplementation
from dwarf.pipeline.results import (
    CaseResult,
    ExecutionResult,
    MetricResult,
    OperationError,
    OperationTiming,
    ResultStatus,
    WorkUnitResult,
)
from dwarf.pipeline.runtime import (
    AttackRuntimeAdapter,
    EmbeddingRuntimeAdapter,
    ErasurePolicy,
    MetricRuntimeAdapter,
    RuntimeAdapterRegistry,
    derive_runtime_seed,
    materialize_watermark,
)


EXPECTED_SERIAL_API = {
    "AttackRuntimeAdapter": AttackRuntimeAdapter,
    "CaseResult": CaseResult,
    "EmbeddingRuntimeAdapter": EmbeddingRuntimeAdapter,
    "ErasurePolicy": ErasurePolicy,
    "ExecutionAbortedError": ExecutionAbortedError,
    "ExecutionError": ExecutionError,
    "ExecutionResult": ExecutionResult,
    "MetricResult": MetricResult,
    "MetricRuntimeAdapter": MetricRuntimeAdapter,
    "OperationError": OperationError,
    "OperationTiming": OperationTiming,
    "Pipeline": PipelineImplementation,
    "ResultStatus": ResultStatus,
    "RuntimeAdapterRegistry": RuntimeAdapterRegistry,
    "SerialExecutor": SerialExecutor,
    "UnsupportedExecutionBackendError": UnsupportedExecutionBackendError,
    "WorkUnitResult": WorkUnitResult,
    "derive_runtime_seed": derive_runtime_seed,
    "materialize_watermark": materialize_watermark,
}


def test_pipeline_is_importable_from_public_package() -> None:
    assert Pipeline is PipelineImplementation


def test_serial_execution_api_is_exported_from_public_package() -> None:
    missing_exports = sorted(set(EXPECTED_SERIAL_API) - set(pipeline_api.__all__))
    assert not missing_exports, f"Missing __all__ entries: {missing_exports}"

    incorrect_attributes = {
        name: getattr(pipeline_api, name, None)
        for name, implementation in EXPECTED_SERIAL_API.items()
        if getattr(pipeline_api, name, None) is not implementation
    }
    assert not incorrect_attributes, (
        "Public package attributes do not reference their implementations: "
        f"{sorted(incorrect_attributes)}"
    )
