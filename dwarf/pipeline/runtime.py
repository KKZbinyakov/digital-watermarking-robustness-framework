# ruff: noqa: UP045
"""Runtime adapters joining canonical pipeline artifacts to DWARF solutions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from typing import Any, Optional

import numpy as np
from PIL import Image

from dwarf.pipeline.artifacts import (
    ExtractedWatermark,
    ImageArtifact,
    OriginalWatermark,
)
from dwarf.pipeline.exceptions import ArtifactValidationError


class ErasurePolicy(str, Enum):
    """How a binary metric handles ``-1`` extracted watermark symbols."""

    COUNT_AS_ERROR = "count_as_error"
    IGNORE = "ignore"
    FAIL = "fail"


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    try:
        return dict(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"expected a parameter mapping, got {type(value).__name__}") from error


def _attribute(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _solution_name(value: Any) -> str:
    name = _attribute(value, "solution_name", "name")
    if not isinstance(name, str) or not name:
        implementation = _attribute(value, "implementation")
        name = getattr(implementation, "__name__", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"could not determine a solution name from {type(value).__name__}")
    return name


def _implementation(value: Any, fallback: Optional[type] = None) -> type:
    implementation = _attribute(value, "implementation", default=fallback)
    if not isinstance(implementation, type):
        raise TypeError(f"could not determine an implementation for {_solution_name(value)!r}")
    return implementation


def _variant_parameters(value: Any, *names: str) -> dict[str, Any]:
    parameters = _attribute(value, *names, default={})
    return _plain_mapping(parameters)


def _looks_like_dimension_error(error: BaseException) -> bool:
    message = str(error).lower()
    fragments = (
        "dimension",
        "dimensions",
        "ndim",
        "shape",
        "2-dimensional",
        "2 dimensional",
        "number of dimensions",
        "buffer has wrong number",
    )
    return any(fragment in message for fragment in fragments)


def _normalise_rgb_result(value: Any, *, operation: str) -> ImageArtifact:
    if isinstance(value, ImageArtifact):
        return value.copy()
    if not isinstance(value, np.ndarray):
        raise ArtifactValidationError(
            f"{operation} must return a numpy.ndarray or ImageArtifact, got {type(value).__name__}"
        )
    if value.ndim != 3 or value.shape[2] != 3:
        raise ArtifactValidationError(f"{operation} returned shape {value.shape}; expected (height, width, 3)")
    if value.dtype != np.uint8:
        value = np.clip(np.rint(value.astype(np.float64)), 0, 255).astype(np.uint8)
    return ImageArtifact(np.ascontiguousarray(value))


def _extract_luma(image: ImageArtifact) -> tuple[np.ndarray, np.ndarray]:
    ycbcr = np.asarray(image.to_pil().convert("YCbCr"), dtype=np.uint8).copy()
    luma = np.ascontiguousarray(ycbcr[..., 0], dtype=np.float64)
    return ycbcr, luma


def _replace_luma(ycbcr: np.ndarray, luma: np.ndarray) -> ImageArtifact:
    if not isinstance(luma, np.ndarray) or luma.ndim != 2:
        raise ArtifactValidationError(
            f"legacy luminance embedding must return a two-dimensional array, got {getattr(luma, 'shape', None)}"
        )
    if luma.shape != ycbcr.shape[:2]:
        raise ArtifactValidationError(
            f"legacy luminance embedding returned shape {luma.shape}, expected {ycbcr.shape[:2]}"
        )
    ycbcr = ycbcr.copy()
    ycbcr[..., 0] = np.clip(np.rint(luma.astype(np.float64)), 0, 255).astype(np.uint8)
    rgb = np.asarray(Image.fromarray(ycbcr, "YCbCr").convert("RGB"), dtype=np.uint8)
    return ImageArtifact(rgb)


def derive_runtime_seed(case_seed: int, domain: str, *parts: Any) -> int:
    """Derive one deterministic 63-bit seed for a runtime sub-operation."""

    if isinstance(case_seed, bool) or not isinstance(case_seed, int) or case_seed < 0:
        raise ValueError("case_seed must be a non-negative integer")
    if not isinstance(domain, str) or not domain:
        raise ValueError("domain must be a non-empty string")
    payload = json.dumps(
        {"case_seed": case_seed, "domain": domain, "parts": parts},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & (2**63 - 1)


def materialize_watermark(reference: Any, config: Any) -> OriginalWatermark:
    """Materialize one fixed or seeded random watermark from a plan reference."""

    kind = str(_attribute(reference, "kind", "type", default=""))
    if "." in kind:
        kind = kind.rsplit(".", 1)[-1]
    kind = kind.lower()

    configured = getattr(config, "watermark", config)
    configured_type = str(getattr(configured, "type", "")).lower()
    if kind in {"fixed", "fixed_bits"} or configured_type == "fixed_bits":
        bits = _attribute(reference, "bits", default=getattr(configured, "bits", None))
        if not isinstance(bits, str):
            raise ArtifactValidationError("fixed watermark reference does not provide a bit string")
        watermark = OriginalWatermark.from_bit_string(bits)
    else:
        length = _attribute(reference, "length", default=getattr(configured, "length", None))
        seed = _attribute(reference, "seed")
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ArtifactValidationError("random watermark reference requires a positive length")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ArtifactValidationError("random watermark reference requires a non-negative seed")
        rng = np.random.default_rng(seed)
        watermark = OriginalWatermark(rng.integers(0, 2, size=length, dtype=np.uint8))

    expected_length = _attribute(reference, "length")
    if expected_length is not None and watermark.length != int(expected_length):
        raise ArtifactValidationError(
            f"materialized watermark length {watermark.length} does not match plan length {expected_length}"
        )
    return watermark


class EmbeddingRuntimeAdapter:
    """Adapter for one embedding/extraction implementation."""

    def embed(
        self,
        variant: Any,
        image: ImageArtifact,
        watermark: OriginalWatermark,
        *,
        seed: Optional[int] = None,
    ) -> ImageArtifact:
        implementation = _implementation(variant)
        parameters = _variant_parameters(
            variant,
            "embedding_parameters",
            "embedding_params",
            "parameters",
        )
        if seed is not None and "seed" not in parameters:
            operation = _attribute(_attribute(variant, "spec"), "operations", default={})
            embedding_spec = operation.get("embedding") if isinstance(operation, Mapping) else None
            accepted = getattr(embedding_spec, "parameters", {})
            if "seed" in accepted:
                parameters["seed"] = seed

        try:
            result = implementation.embedding(
                input_image=image.array.copy(),
                watermark_bits=watermark.mutable_copy(dtype=np.uint8),
                **parameters,
            )
        except (TypeError, ValueError) as direct_error:
            if not _looks_like_dimension_error(direct_error):
                raise
            ycbcr, luma = _extract_luma(image)
            result = implementation.embedding(
                input_image=luma,
                watermark_bits=watermark.mutable_copy(dtype=np.int32),
                **parameters,
            )
            return _replace_luma(ycbcr, np.asarray(result))

        if isinstance(result, np.ndarray) and result.ndim == 2:
            ycbcr, _ = _extract_luma(image)
            return _replace_luma(ycbcr, result)
        return _normalise_rgb_result(result, operation=f"{_solution_name(variant)}.embedding")

    def extract(
        self,
        variant: Any,
        image: ImageArtifact,
        *,
        num_bits: int,
    ) -> ExtractedWatermark:
        implementation = _implementation(variant)
        parameters = _variant_parameters(
            variant,
            "extraction_parameters",
            "extraction_params",
        )
        try:
            result = implementation.extraction(
                input_image=image.array.copy(),
                num_bits=num_bits,
                **parameters,
            )
        except (TypeError, ValueError) as direct_error:
            if not _looks_like_dimension_error(direct_error):
                raise
            _, luma = _extract_luma(image)
            result = implementation.extraction(
                input_image=luma,
                num_bits=num_bits,
                **parameters,
            )
        return ExtractedWatermark(np.asarray(result).reshape(-1))


class AttackRuntimeAdapter:
    """Adapter for canonical RGB attack solutions."""

    def apply(
        self,
        step: Any,
        image: ImageArtifact,
        *,
        seed: Optional[int] = None,
    ) -> ImageArtifact:
        implementation = _implementation(step)
        parameters = _variant_parameters(step, "parameters", "params")
        spec = _attribute(step, "spec")
        operations = getattr(spec, "operations", {}) if spec is not None else {}
        attack_spec = operations.get("attack") if isinstance(operations, Mapping) else None
        accepted = getattr(attack_spec, "parameters", {})
        if seed is not None and "seed" in accepted and "seed" not in parameters:
            parameters["seed"] = seed
        result = implementation.attack(input_image=image.array.copy(), **parameters)
        return _normalise_rgb_result(result, operation=f"{_solution_name(step)}.attack")


class MetricRuntimeAdapter:
    """Adapter converting canonical artifacts to one metric call."""

    def evaluate(
        self,
        metric: Any,
        artifacts: Mapping[str, Any],
        *,
        erasure_policy: ErasurePolicy,
    ) -> float:
        implementation = _implementation(metric)
        parameters = _variant_parameters(metric, "parameters", "params")
        inputs = _plain_mapping(_attribute(metric, "inputs", default={}))
        kwargs = {
            argument: self._metric_value(
                _solution_name(metric),
                artifact_name,
                artifacts[artifact_name],
                artifacts,
                erasure_policy,
            )
            for argument, artifact_name in inputs.items()
        }
        value = implementation.expertise(**kwargs, **parameters)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError(f"{_solution_name(metric)}.expertise must return a number, got {type(value).__name__}")
        value = float(value)
        if np.isnan(value):
            raise ValueError(f"{_solution_name(metric)}.expertise returned NaN")
        return value

    def _metric_value(
        self,
        solution_name: str,
        artifact_name: str,
        value: Any,
        artifacts: Mapping[str, Any],
        erasure_policy: ErasurePolicy,
    ) -> Any:
        if isinstance(value, ImageArtifact):
            return value.array.copy()
        if solution_name == "BER":
            return self._ber_value(artifact_name, value, artifacts, erasure_policy)
        if isinstance(value, OriginalWatermark):
            return value.mutable_copy(dtype=np.uint8)
        if isinstance(value, ExtractedWatermark):
            return value.mutable_copy(dtype=np.int8)
        return value

    def _ber_value(
        self,
        artifact_name: str,
        value: Any,
        artifacts: Mapping[str, Any],
        policy: ErasurePolicy,
    ) -> str:
        if isinstance(value, OriginalWatermark):
            if policy is not ErasurePolicy.IGNORE:
                return value.to_bit_string()
            extracted = artifacts.get("extracted_watermark")
            if not isinstance(extracted, ExtractedWatermark):
                return value.to_bit_string()
            mask = extracted.values != -1
            if not np.any(mask):
                raise ArtifactValidationError("BER is undefined because all extracted bits are erasures")
            return "".join(str(int(item)) for item in value.values[mask])

        if not isinstance(value, ExtractedWatermark):
            raise ArtifactValidationError(f"BER input {artifact_name!r} must be a watermark artifact")
        if not value.has_erasures:
            return value.to_bit_string()
        if policy is ErasurePolicy.FAIL:
            raise ArtifactValidationError(f"BER cannot evaluate {value.erasure_count} erased watermark bits")
        original = artifacts.get("original_watermark")
        if not isinstance(original, OriginalWatermark):
            raise ArtifactValidationError("BER erasure handling requires original_watermark")
        if original.length != value.length:
            raise ArtifactValidationError("BER erasure handling requires equal watermark lengths")
        extracted = value.values.copy()
        if policy is ErasurePolicy.IGNORE:
            mask = extracted != -1
            if not np.any(mask):
                raise ArtifactValidationError("BER is undefined because all extracted bits are erasures")
            extracted = extracted[mask]
        else:
            erased = extracted == -1
            extracted[erased] = 1 - original.values[erased]
        return "".join(str(int(item)) for item in extracted)


class RuntimeAdapterRegistry:
    """Mutable executor-local registry of solution-specific runtime adapters."""

    def __init__(self) -> None:
        self._embeddings: dict[str, EmbeddingRuntimeAdapter] = {}
        self._attacks: dict[str, AttackRuntimeAdapter] = {}
        self._metrics: dict[str, MetricRuntimeAdapter] = {}
        self._default_embedding = EmbeddingRuntimeAdapter()
        self._default_attack = AttackRuntimeAdapter()
        self._default_metric = MetricRuntimeAdapter()

    def register_embedding(self, name: str, adapter: EmbeddingRuntimeAdapter) -> None:
        self._register(self._embeddings, name, adapter, EmbeddingRuntimeAdapter)

    def register_attack(self, name: str, adapter: AttackRuntimeAdapter) -> None:
        self._register(self._attacks, name, adapter, AttackRuntimeAdapter)

    def register_metric(self, name: str, adapter: MetricRuntimeAdapter) -> None:
        self._register(self._metrics, name, adapter, MetricRuntimeAdapter)

    @staticmethod
    def _register(target: dict[str, Any], name: str, adapter: Any, expected: type) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("adapter name must be a non-empty string")
        if not isinstance(adapter, expected):
            raise TypeError(f"adapter must be a {expected.__name__}")
        target[name] = adapter

    def embedding(self, value: Any) -> EmbeddingRuntimeAdapter:
        return self._embeddings.get(_solution_name(value), self._default_embedding)

    def attack(self, value: Any) -> AttackRuntimeAdapter:
        return self._attacks.get(_solution_name(value), self._default_attack)

    def metric(self, value: Any) -> MetricRuntimeAdapter:
        return self._metrics.get(_solution_name(value), self._default_metric)


__all__ = [
    "AttackRuntimeAdapter",
    "EmbeddingRuntimeAdapter",
    "ErasurePolicy",
    "MetricRuntimeAdapter",
    "RuntimeAdapterRegistry",
    "derive_runtime_seed",
    "materialize_watermark",
]
