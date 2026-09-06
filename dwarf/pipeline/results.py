# ruff: noqa: UP045
"""Immutable result records produced by experiment executors."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ResultStatus(str, Enum):
    """Execution status shared by work units, cases and metric evaluations."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class OperationError:
    """Serializable description of one failed pipeline operation."""

    stage: str
    exception_type: str
    message: str

    @classmethod
    def from_exception(cls, stage: str, error: BaseException) -> OperationError:
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a non-empty string")
        return cls(
            stage=stage,
            exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
            message=str(error),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "exception_type": self.exception_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class OperationTiming:
    """Elapsed wall-clock time of one named operation."""

    stage: str
    duration_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("stage must be a non-empty string")
        if isinstance(self.duration_seconds, bool) or not isinstance(self.duration_seconds, (int, float)):
            raise TypeError("duration_seconds must be a number")
        duration = float(self.duration_seconds)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("duration_seconds must be a finite non-negative number")
        object.__setattr__(self, "duration_seconds", duration)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class MetricResult:
    """One metric evaluation at a pipeline checkpoint."""

    metric_id: str
    metric_variant_id: str
    solution_name: str
    checkpoint: str
    work_unit_id: str
    case_id: Optional[str]
    status: ResultStatus
    value: Optional[float]
    parameters: tuple[tuple[str, Any], ...] = ()
    duration_seconds: float = 0.0
    error: Optional[OperationError] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ResultStatus(self.status))
        timing = OperationTiming("metric", self.duration_seconds)
        object.__setattr__(self, "duration_seconds", timing.duration_seconds)
        if self.status is ResultStatus.SUCCESS:
            if self.value is None:
                raise ValueError("a successful metric result requires a value")
            value = float(self.value)
            if math.isnan(value):
                raise ValueError("metric value must not be NaN")
            object.__setattr__(self, "value", value)
            if self.error is not None:
                raise ValueError("a successful metric result must not contain an error")
        elif self.error is None:
            raise ValueError("a non-successful metric result requires an error")

    @property
    def parameter_dict(self) -> dict[str, Any]:
        return dict(self.parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "metric_variant_id": self.metric_variant_id,
            "solution_name": self.solution_name,
            "checkpoint": self.checkpoint,
            "work_unit_id": self.work_unit_id,
            "case_id": self.case_id,
            "status": self.status.value,
            "value": self.value,
            "parameters": self.parameter_dict,
            "duration_seconds": self.duration_seconds,
            "error": None if self.error is None else self.error.to_dict(),
        }


@dataclass(frozen=True)
class CaseResult:
    """Result of one image/embedding/attack/repeat experiment case."""

    case_id: str
    work_unit_id: str
    ordinal: int
    sample_id: str
    embedding_variant_id: str
    attack_variant_id: str
    repeat_index: int
    status: ResultStatus
    metrics: tuple[MetricResult, ...] = ()
    timings: tuple[OperationTiming, ...] = ()
    error: Optional[OperationError] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ResultStatus(self.status))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "timings", tuple(self.timings))
        if self.status in {ResultStatus.FAILED, ResultStatus.SKIPPED} and self.error is None:
            raise ValueError("failed and skipped case results require an error")
        if self.status is ResultStatus.SUCCESS and self.error is not None:
            raise ValueError("a successful case result must not contain an error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "work_unit_id": self.work_unit_id,
            "ordinal": self.ordinal,
            "sample_id": self.sample_id,
            "embedding_variant_id": self.embedding_variant_id,
            "attack_variant_id": self.attack_variant_id,
            "repeat_index": self.repeat_index,
            "status": self.status.value,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "timings": [timing.to_dict() for timing in self.timings],
            "error": None if self.error is None else self.error.to_dict(),
        }


@dataclass(frozen=True)
class WorkUnitResult:
    """Result of one reusable image/watermark/embedding work unit."""

    work_unit_id: str
    sample_id: str
    embedding_variant_id: str
    status: ResultStatus
    cases: tuple[CaseResult, ...]
    metrics: tuple[MetricResult, ...] = ()
    timings: tuple[OperationTiming, ...] = ()
    error: Optional[OperationError] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ResultStatus(self.status))
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "timings", tuple(self.timings))
        if self.status is ResultStatus.FAILED and self.error is None:
            raise ValueError("a failed work unit result requires an error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_unit_id": self.work_unit_id,
            "sample_id": self.sample_id,
            "embedding_variant_id": self.embedding_variant_id,
            "status": self.status.value,
            "cases": [case.to_dict() for case in self.cases],
            "metrics": [metric.to_dict() for metric in self.metrics],
            "timings": [timing.to_dict() for timing in self.timings],
            "error": None if self.error is None else self.error.to_dict(),
        }


@dataclass(frozen=True)
class ExecutionResult:
    """Complete in-memory result of one executor run."""

    plan_fingerprint: str
    catalog_fingerprint: str
    manifest_fingerprint: str
    erasure_policy: str
    work_units: tuple[WorkUnitResult, ...]
    duration_seconds: float
    started_at_utc: str
    finished_at_utc: str
    backend: str = "serial"
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_units", tuple(self.work_units))
        timing = OperationTiming("execution", self.duration_seconds)
        object.__setattr__(self, "duration_seconds", timing.duration_seconds)

    @property
    def cases(self) -> tuple[CaseResult, ...]:
        return tuple(case for unit in self.work_units for case in unit.cases)

    @property
    def metrics(self) -> tuple[MetricResult, ...]:
        return tuple(
            metric
            for unit in self.work_units
            for metric in (*unit.metrics, *(item for case in unit.cases for item in case.metrics))
        )

    def _count_cases(self, status: ResultStatus) -> int:
        return sum(case.status is status for case in self.cases)

    @property
    def successful_case_count(self) -> int:
        return self._count_cases(ResultStatus.SUCCESS)

    @property
    def partial_case_count(self) -> int:
        return self._count_cases(ResultStatus.PARTIAL)

    @property
    def failed_case_count(self) -> int:
        return self._count_cases(ResultStatus.FAILED)

    @property
    def skipped_case_count(self) -> int:
        return self._count_cases(ResultStatus.SKIPPED)

    @property
    def successful_metric_count(self) -> int:
        return sum(metric.status is ResultStatus.SUCCESS for metric in self.metrics)

    @property
    def failed_metric_count(self) -> int:
        return sum(metric.status is ResultStatus.FAILED for metric in self.metrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "plan_fingerprint": self.plan_fingerprint,
            "catalog_fingerprint": self.catalog_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "erasure_policy": self.erasure_policy,
            "duration_seconds": self.duration_seconds,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "metadata": dict(self.metadata),
            "summary": {
                "work_units": len(self.work_units),
                "cases": len(self.cases),
                "successful_cases": self.successful_case_count,
                "partial_cases": self.partial_case_count,
                "failed_cases": self.failed_case_count,
                "skipped_cases": self.skipped_case_count,
                "metrics": len(self.metrics),
                "successful_metrics": self.successful_metric_count,
                "failed_metrics": self.failed_metric_count,
            },
            "work_units": [unit.to_dict() for unit in self.work_units],
        }


__all__ = [
    "CaseResult",
    "ExecutionResult",
    "MetricResult",
    "OperationError",
    "OperationTiming",
    "ResultStatus",
    "WorkUnitResult",
]
