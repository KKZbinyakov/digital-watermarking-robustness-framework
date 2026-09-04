# ruff: noqa: UP007, UP045
"""Immutable metadata describing solutions available to the experiment pipeline.

The core registries intentionally store only ``name -> class`` mappings.  The
pipeline needs a little more information before it can safely plan an
experiment: accepted user parameters, runtime-managed arguments, artifact
bindings, entry points and data contracts.  The descriptors in this module are
pure metadata; constructing them never imports or executes a solution.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CHECKPOINTS = frozenset({"after_embedding", "after_attack", "after_extraction"})


class _UnsetType:
    """Pickle-safe sentinel used for parameters without defaults."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"

    def __reduce__(self):
        return _get_unset, ()


def _get_unset() -> _UnsetType:
    return UNSET


UNSET = _UnsetType()


class FrozenDict(dict):
    """A small pickle-safe immutable mapping used by public descriptors."""

    def _blocked(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("solution metadata mappings are immutable")

    __setitem__ = _blocked
    __delitem__ = _blocked
    __ior__ = _blocked
    clear = _blocked
    pop = _blocked
    popitem = _blocked
    setdefault = _blocked
    update = _blocked

    def __copy__(self) -> FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        return self

    def __reduce__(self):
        return type(self), (dict(self),)


class SolutionKind(str, Enum):
    """Top-level registry category of a solution."""

    ATTACK = "attack"
    EMBEDDING = "embedding"
    METRIC = "metric"


class ParameterKind(str, Enum):
    """Strict value families accepted by user-configurable parameters."""

    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    STRING = "string"
    STRING_SEQUENCE = "string_sequence"
    JSON = "json"


class ArtifactKind(str, Enum):
    """Broad semantic kind of an artifact binding."""

    IMAGE = "image"
    WATERMARK = "watermark"


class DataContract(str, Enum):
    """Named data contracts consumed by future adapters and executors."""

    RGB_UINT8 = "rgb_uint8"
    LUMA_FLOAT64 = "luma_float64"
    BINARY_BITS = "binary_bits"
    TERNARY_BITS = "ternary_bits"
    SCALAR_NUMBER = "scalar_number"


def _validate_name(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a valid Python-style identifier")
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


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if value is UNSET:
        return {"__unset__": True}
    return value


def _fingerprint(value: Any) -> str:
    return json.dumps(
        _plain_json(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _unique_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    result = tuple(values)
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must not contain duplicate values")
    return result


@dataclass(frozen=True)
class ParameterSpec:
    """Validation rules for one user-configurable operation parameter."""

    kind: ParameterKind
    default: Any = UNSET
    required: bool = False
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: tuple[Any, ...] = ()
    allow_none: bool = False
    description: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ParameterKind(self.kind))
        if not isinstance(self.required, bool):
            raise TypeError("required must be a bool")
        if not isinstance(self.allow_none, bool):
            raise TypeError("allow_none must be a bool")
        if self.required and self.default is not UNSET:
            raise ValueError("a required parameter must not declare a default")

        for field_name, bound in (("minimum", self.minimum), ("maximum", self.maximum)):
            if bound is not None and (
                isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(float(bound))
            ):
                raise ValueError(f"{field_name} must be a finite number")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must be less than or equal to maximum")
        if (self.minimum is not None or self.maximum is not None) and self.kind not in {
            ParameterKind.INTEGER,
            ParameterKind.NUMBER,
        }:
            raise ValueError("minimum and maximum are only valid for numeric parameters")

        frozen_choices = tuple(_freeze_json(item, f"choices[{index}]") for index, item in enumerate(self.choices))
        choice_keys = [_fingerprint(item) for item in frozen_choices]
        if len(choice_keys) != len(set(choice_keys)):
            raise ValueError("choices must not contain duplicate values")
        object.__setattr__(self, "choices", frozen_choices)

        if self.default is not UNSET:
            frozen_default = _freeze_json(self.default, "default")
            message = self.validation_error(frozen_default)
            if message is not None:
                raise ValueError(f"invalid default: {message}")
            object.__setattr__(self, "default", frozen_default)

        for index, choice in enumerate(self.choices):
            message = self.validation_error(choice, check_choices=False)
            if message is not None:
                raise ValueError(f"invalid choices[{index}]: {message}")

    @property
    def has_default(self) -> bool:
        return self.default is not UNSET

    def validation_error(self, value: Any, *, check_choices: bool = True) -> Optional[str]:
        """Return a human-readable validation failure, or ``None``."""

        if value is None:
            if self.allow_none:
                return None
            return "must not be null"

        if self.kind is ParameterKind.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                return "must be an integer"
        elif self.kind is ParameterKind.NUMBER:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return "must be a number"
            if not math.isfinite(float(value)):
                return "must be finite"
        elif self.kind is ParameterKind.BOOLEAN:
            if not isinstance(value, bool):
                return "must be a boolean"
        elif self.kind is ParameterKind.STRING:
            if not isinstance(value, str):
                return "must be a string"
        elif self.kind is ParameterKind.STRING_SEQUENCE:
            if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
                return "must be a sequence of strings"
        elif self.kind is ParameterKind.JSON:
            try:
                _freeze_json(value)
            except ValueError as error:
                return str(error)

        if self.kind in {ParameterKind.INTEGER, ParameterKind.NUMBER}:
            numeric = float(value)
            if self.minimum is not None and numeric < self.minimum:
                return f"must be greater than or equal to {self.minimum}"
            if self.maximum is not None and numeric > self.maximum:
                return f"must be less than or equal to {self.maximum}"

        if check_choices and self.choices:
            key = _fingerprint(_freeze_json(value))
            if key not in {_fingerprint(item) for item in self.choices}:
                return f"must be one of {self.choices!r}"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "default": _plain_json(self.default),
            "required": self.required,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "choices": _plain_json(self.choices),
            "allow_none": self.allow_none,
            "description": self.description,
        }


@dataclass(frozen=True)
class ArtifactInputSpec:
    """Allowed pipeline artifacts for one metric argument."""

    artifacts: tuple[str, ...]
    kind: ArtifactKind
    required: bool = True

    def __post_init__(self) -> None:
        artifacts = _unique_tuple(self.artifacts, "artifacts")
        if not artifacts:
            raise ValueError("artifacts must not be empty")
        for artifact in artifacts:
            _validate_name(artifact, "artifact")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "kind", ArtifactKind(self.kind))
        if not isinstance(self.required, bool):
            raise TypeError("required must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": list(self.artifacts),
            "kind": self.kind.value,
            "required": self.required,
        }


@dataclass(frozen=True)
class OperationSpec:
    """Metadata for one callable entry point of a solution."""

    entry_point: str
    parameters: Mapping[str, ParameterSpec] = field(default_factory=dict)
    runtime_arguments: tuple[str, ...] = ()
    artifact_inputs: Mapping[str, ArtifactInputSpec] = field(default_factory=dict)
    checkpoints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_name(self.entry_point, "entry_point")

        parameters = {}
        for name, parameter in self.parameters.items():
            _validate_name(name, "parameter name")
            if not isinstance(parameter, ParameterSpec):
                raise TypeError(f"parameter {name!r} must be a ParameterSpec")
            parameters[name] = parameter

        runtime_arguments = _unique_tuple(self.runtime_arguments, "runtime_arguments")
        for name in runtime_arguments:
            _validate_name(name, "runtime argument")

        artifact_inputs = {}
        for name, artifact in self.artifact_inputs.items():
            _validate_name(name, "artifact input name")
            if not isinstance(artifact, ArtifactInputSpec):
                raise TypeError(f"artifact input {name!r} must be an ArtifactInputSpec")
            artifact_inputs[name] = artifact

        overlap = (set(parameters) & set(runtime_arguments)) | (set(parameters) & set(artifact_inputs))
        if overlap:
            raise ValueError("operation argument groups overlap: " + ", ".join(sorted(overlap)))

        checkpoints = _unique_tuple(self.checkpoints, "checkpoints")
        unknown = sorted(set(checkpoints) - _CHECKPOINTS)
        if unknown:
            raise ValueError("unknown checkpoints: " + ", ".join(unknown))

        object.__setattr__(self, "parameters", FrozenDict(parameters))
        object.__setattr__(self, "runtime_arguments", runtime_arguments)
        object.__setattr__(self, "artifact_inputs", FrozenDict(artifact_inputs))
        object.__setattr__(self, "checkpoints", checkpoints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_point": self.entry_point,
            "parameters": {name: parameter.to_dict() for name, parameter in self.parameters.items()},
            "runtime_arguments": list(self.runtime_arguments),
            "artifact_inputs": {name: item.to_dict() for name, item in self.artifact_inputs.items()},
            "checkpoints": list(self.checkpoints),
        }


@dataclass(frozen=True)
class ParameterLinkSpec:
    """Parameters that must describe the same scalar value or variation axis."""

    source_operation: str
    source_parameter: str
    target_operation: str
    target_parameter: str

    def __post_init__(self) -> None:
        _validate_name(self.source_operation, "source_operation")
        _validate_name(self.source_parameter, "source_parameter")
        _validate_name(self.target_operation, "target_operation")
        _validate_name(self.target_parameter, "target_parameter")

    def to_dict(self) -> dict[str, str]:
        return {
            "source_operation": self.source_operation,
            "source_parameter": self.source_parameter,
            "target_operation": self.target_operation,
            "target_parameter": self.target_parameter,
        }


@dataclass(frozen=True)
class SolutionSpec:
    """Complete pipeline descriptor for one registered solution class."""

    name: str
    kind: SolutionKind
    operations: Mapping[str, OperationSpec]
    contracts: Mapping[str, DataContract] = field(default_factory=dict)
    parameter_links: tuple[ParameterLinkSpec, ...] = ()
    deterministic: bool = True
    resource: str = "cpu"

    def __post_init__(self) -> None:
        _validate_name(self.name, "solution name")
        object.__setattr__(self, "kind", SolutionKind(self.kind))

        operations = {}
        for name, operation in self.operations.items():
            _validate_name(name, "operation name")
            if not isinstance(operation, OperationSpec):
                raise TypeError(f"operation {name!r} must be an OperationSpec")
            operations[name] = operation
        if not operations:
            raise ValueError("a solution must declare at least one operation")

        contracts = {}
        for name, contract in self.contracts.items():
            _validate_name(name, "contract name")
            contracts[name] = DataContract(contract)

        links = tuple(self.parameter_links)
        if any(not isinstance(link, ParameterLinkSpec) for link in links):
            raise TypeError("parameter_links must contain only ParameterLinkSpec values")
        for link in links:
            for operation_name, parameter_name in (
                (link.source_operation, link.source_parameter),
                (link.target_operation, link.target_parameter),
            ):
                if operation_name not in operations:
                    raise ValueError(f"parameter link references unknown operation {operation_name!r}")
                if parameter_name not in operations[operation_name].parameters:
                    raise ValueError(f"parameter link references unknown parameter {operation_name}.{parameter_name}")

        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be a bool")
        if self.resource not in {"cpu", "gpu", "io"}:
            raise ValueError("resource must be 'cpu', 'gpu', or 'io'")

        object.__setattr__(self, "operations", FrozenDict(operations))
        object.__setattr__(self, "contracts", FrozenDict(contracts))
        object.__setattr__(self, "parameter_links", links)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "operations": {name: operation.to_dict() for name, operation in self.operations.items()},
            "contracts": {name: contract.value for name, contract in self.contracts.items()},
            "parameter_links": [link.to_dict() for link in self.parameter_links],
            "deterministic": self.deterministic,
            "resource": self.resource,
        }


__all__ = [
    "UNSET",
    "ArtifactInputSpec",
    "ArtifactKind",
    "DataContract",
    "FrozenDict",
    "OperationSpec",
    "ParameterKind",
    "ParameterLinkSpec",
    "ParameterSpec",
    "SolutionKind",
    "SolutionSpec",
]
