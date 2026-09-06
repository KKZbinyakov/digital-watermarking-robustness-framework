# ruff: noqa: UP045
"""Immutable records emitted by the deterministic experiment planner."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional

from dwarf.pipeline.manifest import SampleReference
from dwarf.pipeline.solution_spec import FrozenDict

PLAN_SCHEMA_VERSION = 1
_MAX_SEED = 2**63 - 1
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINTS = frozenset({"after_embedding", "after_attack", "after_extraction"})
_RESOURCES = frozenset({"cpu", "gpu", "io"})


def _validate_digest(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase hexadecimal SHA-256 digest")
    return value


def _validate_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _validate_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _validate_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _validate_seed(value: int, field_name: str) -> int:
    value = _validate_non_negative_int(value, field_name)
    if value > _MAX_SEED:
        raise ValueError(f"{field_name} must not exceed {_MAX_SEED}")
    return value


def _freeze_json(value: Any, location: str = "value") -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must not contain NaN or infinity")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{location}[{index}]") for index, item in enumerate(value))
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} contains a non-string mapping key: {key!r}")
            frozen[key] = _freeze_json(item, f"{location}.{key}")
        return FrozenDict(frozen)
    raise ValueError(f"{location} contains unsupported value of type {type(value).__name__}")


def _freeze_mapping(value: Mapping[str, Any], field_name: str) -> FrozenDict:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalised = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{field_name} keys must be non-empty strings")
        normalised[key] = _freeze_json(item, f"{field_name}.{key}")
    return FrozenDict({key: normalised[key] for key in sorted(normalised)})


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _mapping_to_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _plain_json(item) for key, item in value.items()}


@dataclass(frozen=True)
class WatermarkReference:
    """Identity and deterministic generation data for one original watermark."""

    watermark_id: str
    kind: Literal["fixed_bits", "random_bits"]
    scope: Literal["fixed_for_run", "per_image", "per_case"]
    length: int
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "watermark_id", _validate_digest(self.watermark_id, "watermark_id"))
        if self.kind not in {"fixed_bits", "random_bits"}:
            raise ValueError("kind must be 'fixed_bits' or 'random_bits'")
        if self.scope not in {"fixed_for_run", "per_image", "per_case"}:
            raise ValueError("scope must be 'fixed_for_run', 'per_image', or 'per_case'")
        object.__setattr__(self, "length", _validate_positive_int(self.length, "length"))

        if self.kind == "fixed_bits":
            if self.scope != "fixed_for_run":
                raise ValueError("fixed-bit watermarks must use fixed_for_run scope")
            if self.seed is not None:
                raise ValueError("fixed-bit watermarks must not declare a seed")
        elif self.seed is None:
            raise ValueError("random-bit watermarks require a deterministic seed")
        else:
            object.__setattr__(self, "seed", _validate_seed(self.seed, "seed"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "watermark_id": self.watermark_id,
            "kind": self.kind,
            "scope": self.scope,
            "length": self.length,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class EmbeddingVariant:
    """One fully resolved embedding and extraction parameter combination."""

    variant_id: str
    declaration_id: str
    solution_name: str
    ordinal: int
    embedding_parameters: Mapping[str, Any]
    extraction_parameters: Mapping[str, Any]
    deterministic: bool
    resource: Literal["cpu", "gpu", "io"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant_id", _validate_digest(self.variant_id, "variant_id"))
        object.__setattr__(self, "declaration_id", _validate_text(self.declaration_id, "declaration_id"))
        object.__setattr__(self, "solution_name", _validate_text(self.solution_name, "solution_name"))
        object.__setattr__(self, "ordinal", _validate_non_negative_int(self.ordinal, "ordinal"))
        object.__setattr__(
            self,
            "embedding_parameters",
            _freeze_mapping(self.embedding_parameters, "embedding_parameters"),
        )
        object.__setattr__(
            self,
            "extraction_parameters",
            _freeze_mapping(self.extraction_parameters, "extraction_parameters"),
        )
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be a boolean")
        if self.resource not in _RESOURCES:
            raise ValueError("resource must be 'cpu', 'gpu', or 'io'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "declaration_id": self.declaration_id,
            "solution_name": self.solution_name,
            "ordinal": self.ordinal,
            "embedding_parameters": _mapping_to_dict(self.embedding_parameters),
            "extraction_parameters": _mapping_to_dict(self.extraction_parameters),
            "deterministic": self.deterministic,
            "resource": self.resource,
        }


@dataclass(frozen=True)
class AttackStepVariant:
    """One parameterised step inside an ordered attack-scenario variant."""

    variant_id: str
    step_index: int
    solution_name: str
    parameter_ordinal: int
    parameters: Mapping[str, Any]
    deterministic: bool
    resource: Literal["cpu", "gpu", "io"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant_id", _validate_digest(self.variant_id, "variant_id"))
        object.__setattr__(self, "step_index", _validate_non_negative_int(self.step_index, "step_index"))
        object.__setattr__(self, "solution_name", _validate_text(self.solution_name, "solution_name"))
        object.__setattr__(
            self,
            "parameter_ordinal",
            _validate_non_negative_int(self.parameter_ordinal, "parameter_ordinal"),
        )
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters, "parameters"))
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be a boolean")
        if self.resource not in _RESOURCES:
            raise ValueError("resource must be 'cpu', 'gpu', or 'io'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "step_index": self.step_index,
            "solution_name": self.solution_name,
            "parameter_ordinal": self.parameter_ordinal,
            "parameters": _mapping_to_dict(self.parameters),
            "deterministic": self.deterministic,
            "resource": self.resource,
        }


@dataclass(frozen=True)
class AttackScenarioVariant:
    """One fully parameterised ordered attack chain, including clean baselines."""

    variant_id: str
    scenario_id: str
    description: Optional[str]
    ordinal: int
    steps: tuple[AttackStepVariant, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant_id", _validate_digest(self.variant_id, "variant_id"))
        object.__setattr__(self, "scenario_id", _validate_text(self.scenario_id, "scenario_id"))
        if self.description is not None and (not isinstance(self.description, str) or not self.description):
            raise ValueError("description must be null or a non-empty string")
        object.__setattr__(self, "ordinal", _validate_non_negative_int(self.ordinal, "ordinal"))
        try:
            steps = tuple(self.steps)
        except TypeError as error:
            raise TypeError("steps must be an iterable of AttackStepVariant values") from error
        if any(not isinstance(step, AttackStepVariant) for step in steps):
            raise TypeError("steps must contain only AttackStepVariant values")
        if tuple(step.step_index for step in steps) != tuple(range(len(steps))):
            raise ValueError("attack steps must have contiguous zero-based indexes")
        object.__setattr__(self, "steps", steps)

    @property
    def is_clean(self) -> bool:
        return not self.steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "scenario_id": self.scenario_id,
            "description": self.description,
            "ordinal": self.ordinal,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class MetricVariant:
    """One metric binding with defaults and parameter variation resolved."""

    variant_id: str
    binding_id: str
    solution_name: str
    checkpoint: Literal["after_embedding", "after_attack", "after_extraction"]
    ordinal: int
    inputs: Mapping[str, str]
    parameters: Mapping[str, Any]
    deterministic: bool
    resource: Literal["cpu", "gpu", "io"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant_id", _validate_digest(self.variant_id, "variant_id"))
        object.__setattr__(self, "binding_id", _validate_text(self.binding_id, "binding_id"))
        object.__setattr__(self, "solution_name", _validate_text(self.solution_name, "solution_name"))
        if self.checkpoint not in _CHECKPOINTS:
            raise ValueError(f"unknown metric checkpoint {self.checkpoint!r}")
        object.__setattr__(self, "ordinal", _validate_non_negative_int(self.ordinal, "ordinal"))
        object.__setattr__(self, "inputs", _freeze_mapping(self.inputs, "inputs"))
        if any(not isinstance(name, str) or not isinstance(artifact, str) for name, artifact in self.inputs.items()):
            raise TypeError("inputs must map strings to strings")
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters, "parameters"))
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be a boolean")
        if self.resource not in _RESOURCES:
            raise ValueError("resource must be 'cpu', 'gpu', or 'io'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "binding_id": self.binding_id,
            "solution_name": self.solution_name,
            "checkpoint": self.checkpoint,
            "ordinal": self.ordinal,
            "inputs": dict(self.inputs),
            "parameters": _mapping_to_dict(self.parameters),
            "deterministic": self.deterministic,
            "resource": self.resource,
        }


@dataclass(frozen=True)
class ExperimentCase:
    """One sample, embedding variant, attack variant and repeat combination."""

    case_id: str
    work_unit_id: str
    ordinal: int
    sample: SampleReference
    embedding: EmbeddingVariant
    attack: AttackScenarioVariant
    repeat_index: int
    embedding_seed: int
    case_seed: int
    watermark: WatermarkReference

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _validate_digest(self.case_id, "case_id"))
        object.__setattr__(self, "work_unit_id", _validate_digest(self.work_unit_id, "work_unit_id"))
        object.__setattr__(self, "ordinal", _validate_non_negative_int(self.ordinal, "ordinal"))
        if not isinstance(self.sample, SampleReference):
            raise TypeError("sample must be a SampleReference")
        if not isinstance(self.embedding, EmbeddingVariant):
            raise TypeError("embedding must be an EmbeddingVariant")
        if not isinstance(self.attack, AttackScenarioVariant):
            raise TypeError("attack must be an AttackScenarioVariant")
        object.__setattr__(
            self,
            "repeat_index",
            _validate_non_negative_int(self.repeat_index, "repeat_index"),
        )
        object.__setattr__(
            self,
            "embedding_seed",
            _validate_seed(self.embedding_seed, "embedding_seed"),
        )
        object.__setattr__(self, "case_seed", _validate_seed(self.case_seed, "case_seed"))
        if not isinstance(self.watermark, WatermarkReference):
            raise TypeError("watermark must be a WatermarkReference")

    def to_dict(self, *, include_absolute_path: bool = True) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "work_unit_id": self.work_unit_id,
            "ordinal": self.ordinal,
            "sample": self.sample.to_dict(include_absolute_path=include_absolute_path),
            "embedding": self.embedding.to_dict(),
            "attack": self.attack.to_dict(),
            "repeat_index": self.repeat_index,
            "embedding_seed": self.embedding_seed,
            "case_seed": self.case_seed,
            "watermark": self.watermark.to_dict(),
        }


@dataclass(frozen=True)
class WorkUnit:
    """Cases that may share one image load and one embedding execution."""

    work_unit_id: str
    ordinal: int
    sample: SampleReference
    embedding: EmbeddingVariant
    repeat_index: int
    embedding_seed: int
    watermark: WatermarkReference
    cases: tuple[ExperimentCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_unit_id", _validate_digest(self.work_unit_id, "work_unit_id"))
        object.__setattr__(self, "ordinal", _validate_non_negative_int(self.ordinal, "ordinal"))
        if not isinstance(self.sample, SampleReference):
            raise TypeError("sample must be a SampleReference")
        if not isinstance(self.embedding, EmbeddingVariant):
            raise TypeError("embedding must be an EmbeddingVariant")
        object.__setattr__(
            self,
            "repeat_index",
            _validate_non_negative_int(self.repeat_index, "repeat_index"),
        )
        object.__setattr__(
            self,
            "embedding_seed",
            _validate_seed(self.embedding_seed, "embedding_seed"),
        )
        if not isinstance(self.watermark, WatermarkReference):
            raise TypeError("watermark must be a WatermarkReference")
        try:
            cases = tuple(self.cases)
        except TypeError as error:
            raise TypeError("cases must be an iterable of ExperimentCase values") from error
        if not cases:
            raise ValueError("a work unit must contain at least one case")
        if any(not isinstance(case, ExperimentCase) for case in cases):
            raise TypeError("cases must contain only ExperimentCase values")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("a work unit must not contain duplicate case IDs")
        for case in cases:
            if case.work_unit_id != self.work_unit_id:
                raise ValueError("every case must reference its containing work unit")
            if case.sample != self.sample:
                raise ValueError("every case must reference the work-unit sample")
            if case.embedding != self.embedding:
                raise ValueError("every case must reference the work-unit embedding variant")
            if case.repeat_index != self.repeat_index:
                raise ValueError("every case must reference the work-unit repeat index")
            if case.embedding_seed != self.embedding_seed:
                raise ValueError("every case must reference the work-unit embedding seed")
            if case.watermark != self.watermark:
                raise ValueError("every case must reference the work-unit watermark")
        object.__setattr__(self, "cases", cases)

    def to_dict(self, *, include_absolute_path: bool = True) -> dict[str, Any]:
        return {
            "work_unit_id": self.work_unit_id,
            "ordinal": self.ordinal,
            "sample": self.sample.to_dict(include_absolute_path=include_absolute_path),
            "embedding": self.embedding.to_dict(),
            "repeat_index": self.repeat_index,
            "embedding_seed": self.embedding_seed,
            "watermark": self.watermark.to_dict(),
            "cases": [case.to_dict(include_absolute_path=include_absolute_path) for case in self.cases],
        }


@dataclass(frozen=True)
class PlanCounts:
    """Pre-computed cardinalities and expected operation invocation counts."""

    sample_count: int
    embedding_variant_count: int
    attack_variant_count: int
    repeat_count: int
    case_count: int
    work_unit_count: int
    embedding_execution_count: int
    attack_execution_count: int
    extraction_execution_count: int
    metric_variant_counts: Mapping[str, int]
    metric_evaluation_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        positive_fields = (
            "sample_count",
            "embedding_variant_count",
            "attack_variant_count",
            "repeat_count",
        )
        for field_name in positive_fields:
            object.__setattr__(
                self,
                field_name,
                _validate_positive_int(getattr(self, field_name), field_name),
            )

        non_negative_fields = (
            "case_count",
            "work_unit_count",
            "embedding_execution_count",
            "attack_execution_count",
            "extraction_execution_count",
        )
        for field_name in non_negative_fields:
            object.__setattr__(
                self,
                field_name,
                _validate_non_negative_int(getattr(self, field_name), field_name),
            )

        object.__setattr__(
            self,
            "metric_variant_counts",
            self._normalise_count_mapping(self.metric_variant_counts, "metric_variant_counts"),
        )
        object.__setattr__(
            self,
            "metric_evaluation_counts",
            self._normalise_count_mapping(self.metric_evaluation_counts, "metric_evaluation_counts"),
        )

    @staticmethod
    def _normalise_count_mapping(value: Mapping[str, int], field_name: str) -> FrozenDict:
        if not isinstance(value, Mapping):
            raise TypeError(f"{field_name} must be a mapping")
        expected = _CHECKPOINTS
        if set(value) != expected:
            raise ValueError(f"{field_name} must contain exactly {sorted(expected)!r}")
        return FrozenDict(
            {
                checkpoint: _validate_non_negative_int(value[checkpoint], f"{field_name}.{checkpoint}")
                for checkpoint in sorted(expected)
            }
        )

    @property
    def metric_variant_count(self) -> int:
        return sum(self.metric_variant_counts.values())

    @property
    def metric_evaluation_count(self) -> int:
        return sum(self.metric_evaluation_counts.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "embedding_variant_count": self.embedding_variant_count,
            "attack_variant_count": self.attack_variant_count,
            "repeat_count": self.repeat_count,
            "case_count": self.case_count,
            "work_unit_count": self.work_unit_count,
            "embedding_execution_count": self.embedding_execution_count,
            "attack_execution_count": self.attack_execution_count,
            "extraction_execution_count": self.extraction_execution_count,
            "metric_variant_count": self.metric_variant_count,
            "metric_variant_counts": dict(self.metric_variant_counts),
            "metric_evaluation_count": self.metric_evaluation_count,
            "metric_evaluation_counts": dict(self.metric_evaluation_counts),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "AttackScenarioVariant",
    "AttackStepVariant",
    "EmbeddingVariant",
    "ExperimentCase",
    "MetricVariant",
    "PlanCounts",
    "WatermarkReference",
    "WorkUnit",
]
