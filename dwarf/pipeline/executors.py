# ruff: noqa: UP045
"""Serial execution of deterministic DWARF experiment plans."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from time import perf_counter
from typing import Any, Optional

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
    ErasurePolicy,
    RuntimeAdapterRegistry,
    derive_runtime_seed,
    materialize_watermark,
)
from dwarf.pipeline.solution_spec import UNSET


class ExecutionError(RuntimeError):
    """Base class for experiment execution failures."""


class ExecutionAbortedError(ExecutionError):
    """Fail-fast execution stopped at the first failed operation."""

    def __init__(self, stage: str, error: BaseException) -> None:
        self.stage = stage
        self.original_error = error
        super().__init__(f"serial execution aborted during {stage}: {error}")


class UnsupportedExecutionBackendError(ExecutionError):
    """The requested execution backend has not been implemented."""


_MISSING = object()


def _attribute(value: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    if default is _MISSING:
        joined = ", ".join(repr(name) for name in names)
        raise AttributeError(f"{type(value).__name__} does not provide any of: {joined}")
    return default


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    return dict(value)


def _identifier(value: Any, *names: str) -> str:
    result = _attribute(value, *names, default=None)
    if result is None:
        return ""
    return str(result)


def _solution_name(value: Any) -> str:
    name = _identifier(value, "solution_name", "name")
    if name:
        return name
    implementation = _attribute(value, "implementation", default=None)
    candidate = getattr(implementation, "__name__", None)
    if isinstance(candidate, str) and candidate:
        return candidate
    raise TypeError(f"could not determine solution name from {type(value).__name__}")


def _parameters(value: Any) -> tuple[tuple[str, Any], ...]:
    mapping = _mapping(_attribute(value, "parameters", "params", default={}))
    return tuple(sorted(mapping.items()))


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _RuntimeMetricVariant:
    metric_id: str
    variant_id: str
    solution_name: str
    implementation: type
    spec: Any
    checkpoint: str
    inputs: Mapping[str, str]
    parameters: Mapping[str, Any]


def _expand_parameter_space(space: Any, operation: Any) -> tuple[dict[str, Any], ...]:
    defaults = {
        name: specification.default
        for name, specification in getattr(operation, "parameters", {}).items()
        if getattr(specification, "default", UNSET) is not UNSET
    }
    fixed = _mapping(getattr(space, "fixed", {}))
    base = {**defaults, **fixed}
    variants = tuple(getattr(space, "variants", ()))
    if variants:
        return tuple({**base, **_mapping(variant)} for variant in variants)

    grid = _mapping(getattr(space, "grid", {}))
    if not grid:
        return (base,)
    names = tuple(sorted(grid))
    return tuple(
        {
            **base,
            **{name: values[index] for index, name in enumerate(names)},
        }
        for values in product(*(tuple(grid[name]) for name in names))
    )


def _resolved_metric_variants(resolved: Any) -> tuple[_RuntimeMetricVariant, ...]:
    variants = []
    for metric in getattr(resolved, "metrics", ()):
        operation = metric.spec.operations["expertise"]
        for parameters in _expand_parameter_space(metric.params, operation):
            variant_id = _fingerprint(
                {
                    "metric_id": metric.id,
                    "solution_name": metric.name,
                    "checkpoint": metric.checkpoint,
                    "inputs": dict(metric.inputs),
                    "parameters": parameters,
                }
            )
            variants.append(
                _RuntimeMetricVariant(
                    metric_id=metric.id,
                    variant_id=variant_id,
                    solution_name=metric.name,
                    implementation=metric.implementation,
                    spec=metric.spec,
                    checkpoint=metric.checkpoint,
                    inputs=metric.inputs,
                    parameters=parameters,
                )
            )
    return tuple(variants)


def _flatten_metric_candidates(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(item for group in value.values() for item in group)
    return tuple(value)


def _plan_metric_variants(plan: Any, resolved: Any) -> tuple[Any, ...]:
    for name in ("metric_variants", "metrics"):
        candidates = _flatten_metric_candidates(getattr(plan, name, None))
        if candidates and all(hasattr(item, "implementation") for item in candidates):
            return candidates
    return _resolved_metric_variants(resolved)


@dataclass(frozen=True)
class _SyntheticWorkUnit:
    work_unit_id: str
    sample: Any
    embedding: Any
    watermark: Any
    embedding_seed: int
    cases: tuple[Any, ...]


def _iter_work_units(plan: Any) -> Iterable[Any]:
    iterator = getattr(plan, "iter_work_units", None)
    if callable(iterator):
        yield from iterator()
        return

    groups: dict[str, list[Any]] = defaultdict(list)
    for case in plan.iter_cases():
        groups[str(case.work_unit_id)].append(case)
    for work_unit_id, cases in groups.items():
        first = cases[0]
        yield _SyntheticWorkUnit(
            work_unit_id=work_unit_id,
            sample=first.sample,
            embedding=first.embedding,
            watermark=first.watermark,
            embedding_seed=int(first.embedding_seed),
            cases=tuple(cases),
        )


def _unit_cases(unit: Any) -> tuple[Any, ...]:
    cases = tuple(_attribute(unit, "cases", default=()))
    if not cases:
        raise ExecutionError(f"work unit {_identifier(unit, 'work_unit_id')!r} contains no cases")
    return cases


def _sample_id(sample: Any) -> str:
    return _identifier(sample, "sample_id", "id", "relative_path")


def _embedding_variant_id(embedding: Any) -> str:
    return _identifier(embedding, "variant_id", "embedding_variant_id", "id")


def _attack_variant_id(case: Any) -> str:
    attack = _attribute(case, "attack", "attack_scenario")
    return _identifier(attack, "variant_id", "attack_variant_id", "id", "scenario_id")


def _metric_variant_id(metric: Any) -> str:
    return _identifier(metric, "variant_id", "metric_variant_id", "id")


def _metric_id(metric: Any) -> str:
    return _identifier(metric, "metric_id", "id")


def _checkpoint(metric: Any) -> str:
    return _identifier(metric, "checkpoint")


def _case_status(metrics: tuple[MetricResult, ...]) -> ResultStatus:
    if any(metric.status is ResultStatus.FAILED for metric in metrics):
        return ResultStatus.PARTIAL
    return ResultStatus.SUCCESS


def _work_unit_status(
    metrics: tuple[MetricResult, ...],
    cases: tuple[CaseResult, ...],
) -> ResultStatus:
    if any(case.status is ResultStatus.FAILED for case in cases):
        return ResultStatus.PARTIAL
    if any(case.status is ResultStatus.PARTIAL for case in cases):
        return ResultStatus.PARTIAL
    if any(metric.status is ResultStatus.FAILED for metric in metrics):
        return ResultStatus.PARTIAL
    return ResultStatus.SUCCESS


@dataclass(frozen=True)
class _ResolvedRuntimeSolution:
    implementation: type
    spec: Any


@dataclass(frozen=True)
class _BoundRuntimeVariant:
    source: Any
    implementation: type
    spec: Any

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)


def _index_resolved_solutions(
    values: Iterable[Any],
    *,
    kind: str,
) -> dict[str, _ResolvedRuntimeSolution]:
    indexed: dict[str, _ResolvedRuntimeSolution] = {}
    for value in values:
        name = _solution_name(value)
        implementation = _attribute(value, "implementation", default=None)
        if not isinstance(implementation, type):
            raise ExecutionError(f"resolved {kind} solution {name!r} does not provide an implementation")
        solution = _ResolvedRuntimeSolution(
            implementation=implementation,
            spec=_attribute(value, "spec", default=None),
        )
        previous = indexed.get(name)
        if previous is not None and previous.implementation is not implementation:
            raise ExecutionError(f"resolved {kind} solution {name!r} maps to multiple implementations")
        indexed[name] = solution
    return indexed


def _resolved_embedding_solutions(resolved: Any) -> dict[str, _ResolvedRuntimeSolution]:
    return _index_resolved_solutions(
        getattr(resolved, "embeddings", ()),
        kind="embedding",
    )


def _resolved_attack_solutions(resolved: Any) -> dict[str, _ResolvedRuntimeSolution]:
    steps = (step for scenario in getattr(resolved, "attack_scenarios", ()) for step in getattr(scenario, "steps", ()))
    return _index_resolved_solutions(steps, kind="attack")


def _bind_runtime_variant(
    value: Any,
    solutions: Mapping[str, _ResolvedRuntimeSolution],
    *,
    kind: str,
) -> Any:
    if isinstance(_attribute(value, "implementation", default=None), type):
        return value

    name = _solution_name(value)
    solution = solutions.get(name)
    if solution is None:
        raise ExecutionError(f"resolved experiment does not provide an implementation for {kind} solution {name!r}")
    return _BoundRuntimeVariant(
        source=value,
        implementation=solution.implementation,
        spec=solution.spec,
    )


class SerialExecutor:
    """Execute work units serially in the current Python process."""

    def __init__(
        self,
        plan: Any,
        resolved: Any,
        dataset_source: Any,
        *,
        adapters: Optional[RuntimeAdapterRegistry] = None,
        erasure_policy: ErasurePolicy = ErasurePolicy.COUNT_AS_ERROR,
        verify_integrity: bool = True,
        fail_fast: Optional[bool] = None,
    ) -> None:
        self.plan = plan
        self.resolved = resolved
        self.dataset_source = dataset_source
        self.adapters = RuntimeAdapterRegistry() if adapters is None else adapters
        self.erasure_policy = ErasurePolicy(erasure_policy)
        if not isinstance(verify_integrity, bool):
            raise TypeError("verify_integrity must be a boolean")
        self.verify_integrity = verify_integrity
        configured_fail_fast = bool(resolved.config.execution.fail_fast)
        self.fail_fast = configured_fail_fast if fail_fast is None else fail_fast
        if not isinstance(self.fail_fast, bool):
            raise TypeError("fail_fast must be a boolean")
        self._embedding_solutions = _resolved_embedding_solutions(resolved)
        self._attack_solutions = _resolved_attack_solutions(resolved)
        self.metric_variants = _plan_metric_variants(plan, resolved)
        self.metrics_by_checkpoint = {
            checkpoint: tuple(metric for metric in self.metric_variants if _checkpoint(metric) == checkpoint)
            for checkpoint in ("after_embedding", "after_attack", "after_extraction")
        }

    def execute(self) -> ExecutionResult:
        """Execute the entire plan and return immutable in-memory results."""

        started = datetime.now(timezone.utc)
        started_clock = perf_counter()
        work_unit_results = tuple(self._execute_work_unit(unit) for unit in _iter_work_units(self.plan))
        finished = datetime.now(timezone.utc)
        return ExecutionResult(
            plan_fingerprint=str(getattr(self.plan, "fingerprint", "")),
            catalog_fingerprint=str(self.resolved.catalog_fingerprint),
            manifest_fingerprint=str(getattr(self.plan.manifest, "fingerprint", "")),
            erasure_policy=self.erasure_policy.value,
            work_units=work_unit_results,
            duration_seconds=perf_counter() - started_clock,
            started_at_utc=started.isoformat(),
            finished_at_utc=finished.isoformat(),
            metadata=(
                ("declared_cases", int(getattr(self.plan.counts, "case_count", len(work_unit_results)))),
                ("declared_work_units", int(getattr(self.plan.counts, "work_unit_count", len(work_unit_results)))),
            ),
        )

    run = execute

    def _execute_work_unit(self, unit: Any) -> WorkUnitResult:
        cases = _unit_cases(unit)
        work_unit_id = _identifier(unit, "work_unit_id")
        sample = _attribute(unit, "sample")
        embedding = _attribute(unit, "embedding")
        runtime_embedding = _bind_runtime_variant(
            embedding,
            self._embedding_solutions,
            kind="embedding",
        )
        watermark_reference = _attribute(unit, "watermark")
        embedding_seed = int(_attribute(unit, "embedding_seed", default=_attribute(cases[0], "embedding_seed")))
        timings = []

        try:
            start = perf_counter()
            original = self.dataset_source.load(
                sample,
                verify_integrity=self.verify_integrity,
            )
            timings.append(OperationTiming("load_image", perf_counter() - start))

            start = perf_counter()
            watermark = materialize_watermark(watermark_reference, self.resolved.config)
            timings.append(OperationTiming("materialize_watermark", perf_counter() - start))

            start = perf_counter()
            embedded = self.adapters.embedding(runtime_embedding).embed(
                runtime_embedding,
                original,
                watermark,
                seed=embedding_seed,
            )
            timings.append(OperationTiming("embedding", perf_counter() - start))
        except Exception as error:
            return self._failed_work_unit(unit, cases, error, timings)

        artifacts = {
            "original_image": original,
            "embedded_image": embedded,
            "original_watermark": watermark,
        }
        unit_metrics = self._evaluate_metrics(
            self.metrics_by_checkpoint["after_embedding"],
            artifacts,
            work_unit_id=work_unit_id,
            case_id=None,
        )
        case_results = tuple(
            self._execute_case(
                case,
                runtime_embedding,
                embedded,
                original,
                watermark,
            )
            for case in cases
        )
        return WorkUnitResult(
            work_unit_id=work_unit_id,
            sample_id=_sample_id(sample),
            embedding_variant_id=_embedding_variant_id(embedding),
            status=_work_unit_status(unit_metrics, case_results),
            cases=case_results,
            metrics=unit_metrics,
            timings=tuple(timings),
        )

    def _failed_work_unit(
        self,
        unit: Any,
        cases: tuple[Any, ...],
        error: BaseException,
        timings: list[OperationTiming],
    ) -> WorkUnitResult:
        stage = self._failure_stage(timings)
        if self.fail_fast:
            raise ExecutionAbortedError(stage, error) from error
        operation_error = OperationError.from_exception(stage, error)
        sample = _attribute(unit, "sample")
        embedding = _attribute(unit, "embedding")
        skipped = tuple(
            CaseResult(
                case_id=str(case.case_id),
                work_unit_id=str(case.work_unit_id),
                ordinal=int(case.ordinal),
                sample_id=_sample_id(case.sample),
                embedding_variant_id=_embedding_variant_id(case.embedding),
                attack_variant_id=_attack_variant_id(case),
                repeat_index=int(case.repeat_index),
                status=ResultStatus.SKIPPED,
                error=operation_error,
            )
            for case in cases
        )
        return WorkUnitResult(
            work_unit_id=_identifier(unit, "work_unit_id"),
            sample_id=_sample_id(sample),
            embedding_variant_id=_embedding_variant_id(embedding),
            status=ResultStatus.FAILED,
            cases=skipped,
            timings=tuple(timings),
            error=operation_error,
        )

    @staticmethod
    def _failure_stage(timings: list[OperationTiming]) -> str:
        completed = {timing.stage for timing in timings}
        if "load_image" not in completed:
            return "load_image"
        if "materialize_watermark" not in completed:
            return "materialize_watermark"
        return "embedding"

    def _execute_case(
        self,
        case: Any,
        runtime_embedding: Any,
        embedded: Any,
        original: Any,
        watermark: Any,
    ) -> CaseResult:
        case_id = str(case.case_id)
        work_unit_id = str(case.work_unit_id)
        timings = []
        metrics = []
        attacked = embedded.copy()

        try:
            for step_index, step in enumerate(case.attack.steps):
                runtime_step = _bind_runtime_variant(
                    step,
                    self._attack_solutions,
                    kind="attack",
                )
                start = perf_counter()
                seed = derive_runtime_seed(
                    int(case.case_seed),
                    "attack",
                    step_index,
                    _identifier(step, "variant_id", "id"),
                    _solution_name(step),
                )
                attacked = self.adapters.attack(runtime_step).apply(
                    runtime_step,
                    attacked,
                    seed=seed,
                )
                timings.append(
                    OperationTiming(
                        f"attack[{step_index}]:{_solution_name(step)}",
                        perf_counter() - start,
                    )
                )
        except Exception as error:
            return self._failed_case(case, "attack", error, metrics, timings)

        artifacts = {
            "original_image": original,
            "embedded_image": embedded,
            "attacked_image": attacked,
            "original_watermark": watermark,
        }
        metrics.extend(
            self._evaluate_metrics(
                self.metrics_by_checkpoint["after_attack"],
                artifacts,
                work_unit_id=work_unit_id,
                case_id=case_id,
            )
        )

        try:
            start = perf_counter()
            extracted = self.adapters.embedding(runtime_embedding).extract(
                runtime_embedding,
                attacked,
                num_bits=watermark.length,
            )
            timings.append(OperationTiming("extraction", perf_counter() - start))
        except Exception as error:
            return self._failed_case(case, "extraction", error, metrics, timings)

        artifacts["extracted_watermark"] = extracted
        metrics.extend(
            self._evaluate_metrics(
                self.metrics_by_checkpoint["after_extraction"],
                artifacts,
                work_unit_id=work_unit_id,
                case_id=case_id,
            )
        )
        metric_results = tuple(metrics)
        return CaseResult(
            case_id=case_id,
            work_unit_id=work_unit_id,
            ordinal=int(case.ordinal),
            sample_id=_sample_id(case.sample),
            embedding_variant_id=_embedding_variant_id(case.embedding),
            attack_variant_id=_attack_variant_id(case),
            repeat_index=int(case.repeat_index),
            status=_case_status(metric_results),
            metrics=metric_results,
            timings=tuple(timings),
        )

    def _failed_case(
        self,
        case: Any,
        stage: str,
        error: BaseException,
        metrics: list[MetricResult],
        timings: list[OperationTiming],
    ) -> CaseResult:
        if self.fail_fast:
            raise ExecutionAbortedError(stage, error) from error
        operation_error = OperationError.from_exception(stage, error)
        return CaseResult(
            case_id=str(case.case_id),
            work_unit_id=str(case.work_unit_id),
            ordinal=int(case.ordinal),
            sample_id=_sample_id(case.sample),
            embedding_variant_id=_embedding_variant_id(case.embedding),
            attack_variant_id=_attack_variant_id(case),
            repeat_index=int(case.repeat_index),
            status=ResultStatus.FAILED,
            metrics=tuple(metrics),
            timings=tuple(timings),
            error=operation_error,
        )

    def _evaluate_metrics(
        self,
        metrics: tuple[Any, ...],
        artifacts: Mapping[str, Any],
        *,
        work_unit_id: str,
        case_id: Optional[str],
    ) -> tuple[MetricResult, ...]:
        results = []
        for metric in metrics:
            start = perf_counter()
            try:
                value = self.adapters.metric(metric).evaluate(
                    metric,
                    artifacts,
                    erasure_policy=self.erasure_policy,
                )
            except Exception as error:
                if self.fail_fast:
                    raise ExecutionAbortedError(
                        f"metric:{_metric_id(metric) or _solution_name(metric)}",
                        error,
                    ) from error
                results.append(
                    MetricResult(
                        metric_id=_metric_id(metric),
                        metric_variant_id=_metric_variant_id(metric),
                        solution_name=_solution_name(metric),
                        checkpoint=_checkpoint(metric),
                        work_unit_id=work_unit_id,
                        case_id=case_id,
                        status=ResultStatus.FAILED,
                        value=None,
                        parameters=_parameters(metric),
                        duration_seconds=perf_counter() - start,
                        error=OperationError.from_exception("metric", error),
                    )
                )
            else:
                results.append(
                    MetricResult(
                        metric_id=_metric_id(metric),
                        metric_variant_id=_metric_variant_id(metric),
                        solution_name=_solution_name(metric),
                        checkpoint=_checkpoint(metric),
                        work_unit_id=work_unit_id,
                        case_id=case_id,
                        status=ResultStatus.SUCCESS,
                        value=value,
                        parameters=_parameters(metric),
                        duration_seconds=perf_counter() - start,
                    )
                )
        return tuple(results)


__all__ = [
    "ExecutionAbortedError",
    "ExecutionError",
    "SerialExecutor",
    "UnsupportedExecutionBackendError",
]
