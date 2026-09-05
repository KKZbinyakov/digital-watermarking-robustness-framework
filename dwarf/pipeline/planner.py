# ruff: noqa: B905, UP045
"""Deterministic, lazy expansion of validated DWARF experiment declarations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, Optional

from dwarf.pipeline.config import (
    FixedBitsWatermarkSpec,
    ParameterSpace,
    RandomBitsWatermarkSpec,
)
from dwarf.pipeline.exceptions import PlanningError, PlanTooLargeError
from dwarf.pipeline.manifest import DatasetManifest, SampleReference
from dwarf.pipeline.plan import (
    PLAN_SCHEMA_VERSION,
    AttackScenarioVariant,
    AttackStepVariant,
    EmbeddingVariant,
    ExperimentCase,
    MetricVariant,
    PlanCounts,
    WatermarkReference,
    WorkUnit,
)
from dwarf.pipeline.resolved import (
    ResolvedAttackScenario,
    ResolvedAttackStep,
    ResolvedEmbedding,
    ResolvedExperiment,
)
from dwarf.pipeline.solution_spec import (
    UNSET,
    FrozenDict,
    OperationSpec,
    ParameterKind,
    ParameterSpec,
    SolutionKind,
    SolutionSpec,
)

_MAX_SEED = 2**63 - 1
_CHECKPOINTS = ("after_embedding", "after_attack", "after_extraction")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _plain_json(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _stable_digest(domain: str, value: Any) -> str:
    payload = {
        "domain": domain,
        "value": _plain_json(value),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def derive_stable_seed(domain: str, *parts: Any) -> int:
    """Derive a portable non-negative 63-bit seed from canonical JSON values."""

    if not isinstance(domain, str) or not domain:
        raise ValueError("domain must be a non-empty string")
    digest = hashlib.sha256(_canonical_json({"domain": domain, "parts": list(parts)}).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & _MAX_SEED


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PlanningError("parameter values must not contain NaN or infinity")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, Mapping):
        normalised = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PlanningError("parameter mappings must use string keys")
            normalised[key] = _freeze_json(item)
        return FrozenDict({key: normalised[key] for key in sorted(normalised)})
    raise PlanningError(f"parameter value has unsupported type {type(value).__name__}")


def _normalise_parameter_value(value: Any, parameter: ParameterSpec, location: str) -> Any:
    message = parameter.validation_error(value)
    if message is not None:
        raise PlanningError(f"{location}: parameter value {message}")
    if value is None:
        return None
    if parameter.kind is ParameterKind.NUMBER:
        return float(value)
    if parameter.kind is ParameterKind.INTEGER:
        return int(value)
    if parameter.kind is ParameterKind.BOOLEAN:
        return bool(value)
    if parameter.kind is ParameterKind.STRING:
        return str(value)
    if parameter.kind is ParameterKind.STRING_SEQUENCE:
        return tuple(value)
    return _freeze_json(value)


def _solution_identity(kind: SolutionKind, implementation: type, spec: SolutionSpec) -> dict[str, Any]:
    return {
        "kind": kind.value,
        "name": spec.name,
        "implementation": f"{implementation.__module__}.{implementation.__qualname__}",
        "spec": spec.to_dict(),
    }


@dataclass(frozen=True)
class _Dimension:
    """One finite variation axis over namespaced operation parameters."""

    name: str
    options: tuple[Mapping[tuple[str, str], Any], ...]

    @property
    def count(self) -> int:
        return len(self.options)


@dataclass(frozen=True)
class _OperationBlueprint:
    base: Mapping[tuple[str, str], Any]
    dimensions: tuple[_Dimension, ...]
    parameter_dimensions: Mapping[tuple[str, str], int]

    @property
    def count(self) -> int:
        result = 1
        for dimension in self.dimensions:
            result *= dimension.count
        return result


def _build_operation_blueprint(
    namespace: str,
    space: ParameterSpace,
    operation: OperationSpec,
) -> _OperationBlueprint:
    base: dict[tuple[str, str], Any] = {}
    dimensions: list[_Dimension] = []
    parameter_dimensions: dict[tuple[str, str], int] = {}

    variable_parameter_names = set(space.grid)
    if space.variants:
        variable_parameter_names.update(space.variants[0])
    configured_parameter_names = set(space.fixed) | variable_parameter_names

    for parameter_name in sorted(operation.parameters):
        parameter = operation.parameters[parameter_name]
        if parameter.default is not UNSET and parameter_name not in configured_parameter_names:
            base[(namespace, parameter_name)] = _normalise_parameter_value(
                parameter.default,
                parameter,
                f"{namespace}.{parameter_name}.default",
            )

    for parameter_name in sorted(space.fixed):
        parameter = operation.parameters.get(parameter_name)
        if parameter is None:
            raise PlanningError(
                f"{namespace}.params.fixed.{parameter_name}: parameter is absent from solution metadata"
            )
        base[(namespace, parameter_name)] = _normalise_parameter_value(
            space.fixed[parameter_name],
            parameter,
            f"{namespace}.params.fixed.{parameter_name}",
        )

    for parameter_name in sorted(space.grid):
        parameter = operation.parameters.get(parameter_name)
        if parameter is None:
            raise PlanningError(f"{namespace}.params.grid.{parameter_name}: parameter is absent from solution metadata")
        key = (namespace, parameter_name)
        options = tuple(
            {
                key: _normalise_parameter_value(
                    value,
                    parameter,
                    f"{namespace}.params.grid.{parameter_name}[{index}]",
                )
            }
            for index, value in enumerate(space.grid[parameter_name])
        )
        parameter_dimensions[key] = len(dimensions)
        dimensions.append(_Dimension(f"{namespace}.grid.{parameter_name}", options))

    if space.variants:
        variant_keys = tuple(sorted(space.variants[0]))
        options = []
        for variant_index, variant in enumerate(space.variants):
            option = {}
            for parameter_name in variant_keys:
                parameter = operation.parameters.get(parameter_name)
                if parameter is None:
                    raise PlanningError(
                        f"{namespace}.params.variants[{variant_index}].{parameter_name}: "
                        "parameter is absent from solution metadata"
                    )
                key = (namespace, parameter_name)
                option[key] = _normalise_parameter_value(
                    variant[parameter_name],
                    parameter,
                    f"{namespace}.params.variants[{variant_index}].{parameter_name}",
                )
            options.append(option)
        dimension_index = len(dimensions)
        for parameter_name in variant_keys:
            parameter_dimensions[(namespace, parameter_name)] = dimension_index
        dimensions.append(_Dimension(f"{namespace}.variants", tuple(options)))

    for parameter_name, parameter in operation.parameters.items():
        key = (namespace, parameter_name)
        if parameter.required and key not in base and key not in parameter_dimensions:
            raise PlanningError(f"{namespace}: required parameter {parameter_name!r} is unresolved")

    return _OperationBlueprint(
        base=base,
        dimensions=tuple(dimensions),
        parameter_dimensions=parameter_dimensions,
    )


def _merge_assignments(*assignments: Mapping[tuple[str, str], Any]) -> dict[tuple[str, str], Any]:
    merged: dict[tuple[str, str], Any] = {}
    for assignment in assignments:
        for key, value in assignment.items():
            if key in merged and _canonical_json(merged[key]) != _canonical_json(value):
                raise PlanningError(f"variation dimensions assign conflicting values to {key[0]}.{key[1]}")
            merged[key] = value
    return merged


def _iter_dimension_assignments(dimensions: Sequence[_Dimension]) -> Iterator[dict[tuple[str, str], Any]]:
    if not dimensions:
        yield {}
        return
    option_sets = tuple(dimension.options for dimension in dimensions)
    for selected in product(*option_sets):
        yield _merge_assignments(*selected)


def _parameter_from_blueprint(
    blueprint: _OperationBlueprint,
    key: tuple[str, str],
) -> tuple[Any, tuple[Any, ...], bool]:
    dimension_index = blueprint.parameter_dimensions.get(key)
    if dimension_index is None:
        return blueprint.base.get(key, UNSET), (), False
    dimension = blueprint.dimensions[dimension_index]
    values = tuple(option[key] for option in dimension.options)
    varying = len({_canonical_json(value) for value in values}) > 1
    return UNSET, values, varying


def _joint_embedding_blueprint(
    embedding: ResolvedEmbedding,
) -> tuple[Mapping[tuple[str, str], Any], tuple[_Dimension, ...]]:
    embedding_blueprint = _build_operation_blueprint(
        "embedding",
        embedding.embedding.params,
        embedding.spec.operations["embedding"],
    )
    extraction_blueprint = _build_operation_blueprint(
        "extraction",
        embedding.extraction.params,
        embedding.spec.operations["extraction"],
    )

    base = _merge_assignments(embedding_blueprint.base, extraction_blueprint.base)
    dimensions = embedding_blueprint.dimensions + extraction_blueprint.dimensions
    offset = len(embedding_blueprint.dimensions)
    parameter_dimensions = dict(embedding_blueprint.parameter_dimensions)
    parameter_dimensions.update(
        {key: dimension_index + offset for key, dimension_index in extraction_blueprint.parameter_dimensions.items()}
    )

    parent = list(range(len(dimensions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    combined_blueprint = _OperationBlueprint(base, dimensions, parameter_dimensions)
    for link in embedding.spec.parameter_links:
        source_key = (link.source_operation, link.source_parameter)
        target_key = (link.target_operation, link.target_parameter)
        source_scalar, source_values, source_varying = _parameter_from_blueprint(combined_blueprint, source_key)
        target_scalar, target_values, target_varying = _parameter_from_blueprint(combined_blueprint, target_key)

        if source_varying != target_varying:
            raise PlanningError(
                f"linked parameters {source_key[0]}.{source_key[1]} and "
                f"{target_key[0]}.{target_key[1]} do not expose compatible variation axes"
            )
        if source_varying:
            source_dimension = parameter_dimensions[source_key]
            target_dimension = parameter_dimensions[target_key]
            if len(source_values) != len(target_values):
                raise PlanningError(
                    f"linked parameters {source_key[0]}.{source_key[1]} and "
                    f"{target_key[0]}.{target_key[1]} have different axis lengths"
                )
            for source_value, target_value in zip(source_values, target_values):
                if _canonical_json(source_value) != _canonical_json(target_value):
                    raise PlanningError(
                        f"linked parameters {source_key[0]}.{source_key[1]} and "
                        f"{target_key[0]}.{target_key[1]} have different ordered values"
                    )
            union(source_dimension, target_dimension)
            continue

        if source_values:
            source_scalar = source_values[0]
        if target_values:
            target_scalar = target_values[0]
        if source_scalar is UNSET or target_scalar is UNSET:
            raise PlanningError(
                f"linked parameters {source_key[0]}.{source_key[1]} and {target_key[0]}.{target_key[1]} are unresolved"
            )
        if _canonical_json(source_scalar) != _canonical_json(target_scalar):
            raise PlanningError(
                f"linked parameters {source_key[0]}.{source_key[1]} and "
                f"{target_key[0]}.{target_key[1]} have different scalar values"
            )

    components: dict[int, list[int]] = {}
    for dimension_index in range(len(dimensions)):
        components.setdefault(find(dimension_index), []).append(dimension_index)

    merged_dimensions = []
    for component_indexes in sorted(components.values(), key=min):
        component_dimensions = [dimensions[index] for index in component_indexes]
        component_count = component_dimensions[0].count
        if any(dimension.count != component_count for dimension in component_dimensions[1:]):
            names = ", ".join(dimension.name for dimension in component_dimensions)
            raise PlanningError(f"linked variation dimensions have incompatible lengths: {names}")
        options = tuple(
            _merge_assignments(*(dimension.options[index] for dimension in component_dimensions))
            for index in range(component_count)
        )
        merged_dimensions.append(
            _Dimension(
                "+".join(dimension.name for dimension in component_dimensions),
                options,
            )
        )

    return base, tuple(merged_dimensions)


def _embedding_variant_count(embedding: ResolvedEmbedding) -> int:
    _, dimensions = _joint_embedding_blueprint(embedding)
    count = 1
    for dimension in dimensions:
        count *= dimension.count
    return count


def _iter_embedding_parameter_pairs(
    embedding: ResolvedEmbedding,
) -> Iterator[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    base, dimensions = _joint_embedding_blueprint(embedding)
    for selected in _iter_dimension_assignments(dimensions):
        values = _merge_assignments(base, selected)
        embedding_parameters = FrozenDict(
            {
                parameter_name: value
                for (operation_name, parameter_name), value in sorted(values.items())
                if operation_name == "embedding"
            }
        )
        extraction_parameters = FrozenDict(
            {
                parameter_name: value
                for (operation_name, parameter_name), value in sorted(values.items())
                if operation_name == "extraction"
            }
        )
        yield embedding_parameters, extraction_parameters


def _operation_variant_count(space: ParameterSpace, operation: OperationSpec) -> int:
    return _build_operation_blueprint("operation", space, operation).count


def _iter_operation_parameters(
    space: ParameterSpace,
    operation: OperationSpec,
) -> Iterator[Mapping[str, Any]]:
    blueprint = _build_operation_blueprint("operation", space, operation)
    for selected in _iter_dimension_assignments(blueprint.dimensions):
        values = _merge_assignments(blueprint.base, selected)
        yield FrozenDict({parameter_name: value for (_, parameter_name), value in sorted(values.items())})


def _attack_scenario_variant_count(scenario: ResolvedAttackScenario) -> int:
    count = 1
    for step in scenario.steps:
        count *= _operation_variant_count(step.params, step.spec.operations["attack"])
    return count


def _metric_variant_counts(resolved: ResolvedExperiment) -> dict[str, int]:
    counts = dict.fromkeys(_CHECKPOINTS, 0)
    for metric in resolved.metrics:
        counts[metric.checkpoint] += _operation_variant_count(metric.params, metric.spec.operations["expertise"])
    return counts


def _parameter_space_payload(space: ParameterSpace) -> dict[str, Any]:
    return {
        "fixed": _plain_json(space.fixed),
        "grid": _plain_json(space.grid),
        "variants": _plain_json(space.variants),
    }


def _plan_fingerprint_payload(resolved: ResolvedExperiment, manifest: DatasetManifest) -> dict[str, Any]:
    config = resolved.config
    return {
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "config_schema_version": config.schema_version,
        "manifest_fingerprint": manifest.fingerprint,
        "dataset_preprocessing": config.dataset.preprocessing.model_dump(mode="json"),
        "seed": config.experiment.seed,
        "repeats": config.experiment.repeats,
        "watermark": config.watermark.model_dump(mode="json"),
        "plugins": list(resolved.plugins),
        "embeddings": [
            {
                "id": item.id,
                "solution": _solution_identity(SolutionKind.EMBEDDING, item.implementation, item.spec),
                "embedding": _parameter_space_payload(item.embedding.params),
                "extraction": _parameter_space_payload(item.extraction.params),
            }
            for item in resolved.embeddings
        ],
        "attack_scenarios": [
            {
                "id": scenario.id,
                "steps": [
                    {
                        "index": step.index,
                        "solution": _solution_identity(SolutionKind.ATTACK, step.implementation, step.spec),
                        "params": _parameter_space_payload(step.params),
                    }
                    for step in scenario.steps
                ],
            }
            for scenario in resolved.attack_scenarios
        ],
        "metrics": [
            {
                "id": metric.id,
                "solution": _solution_identity(SolutionKind.METRIC, metric.implementation, metric.spec),
                "checkpoint": metric.checkpoint,
                "inputs": dict(metric.inputs),
                "params": _parameter_space_payload(metric.params),
            }
            for metric in resolved.metrics
        ],
    }


def _case_context_id(resolved: ResolvedExperiment) -> str:
    config = resolved.config
    payload = {
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "config_schema_version": config.schema_version,
        "dataset_preprocessing": config.dataset.preprocessing.model_dump(mode="json"),
        "seed": config.experiment.seed,
        "watermark": config.watermark.model_dump(mode="json"),
    }
    return _stable_digest("case-context", payload)


def _validate_solution_binding(
    *,
    kind: SolutionKind,
    name: str,
    implementation: type,
    spec: SolutionSpec,
    location: str,
) -> None:
    if not isinstance(implementation, type):
        raise PlanningError(f"{location}.implementation must be a class")
    if not isinstance(spec, SolutionSpec):
        raise PlanningError(f"{location}.spec must be a SolutionSpec")
    if implementation.__name__ != name:
        raise PlanningError(f"{location}.implementation is {implementation.__name__!r}, expected {name!r}")
    if spec.kind is not kind:
        raise PlanningError(f"{location}.spec declares {spec.kind.value!r}, expected {kind.value!r}")
    if spec.name != name:
        raise PlanningError(f"{location}.spec describes solution {spec.name!r}, expected {name!r}")


def _validate_resolved_alignment(resolved: ResolvedExperiment) -> None:
    config = resolved.config
    if tuple(resolved.plugins) != tuple(config.plugins):
        raise PlanningError("resolved plugins do not match the experiment configuration")

    if len(resolved.embeddings) != len(config.embeddings):
        raise PlanningError("resolved embedding declarations do not match the experiment configuration")
    for index, (resolved_item, configured_item) in enumerate(zip(resolved.embeddings, config.embeddings)):
        location = f"resolved.embeddings[{index}]"
        if (resolved_item.id, resolved_item.name) != (configured_item.id, configured_item.name):
            raise PlanningError(f"{location} does not match the configuration")
        if resolved_item.embedding != configured_item.embedding:
            raise PlanningError(f"{location}.embedding does not match the configuration")
        if resolved_item.extraction != configured_item.extraction:
            raise PlanningError(f"{location}.extraction does not match the configuration")
        _validate_solution_binding(
            kind=SolutionKind.EMBEDDING,
            name=resolved_item.name,
            implementation=resolved_item.implementation,
            spec=resolved_item.spec,
            location=location,
        )

    if len(resolved.attack_scenarios) != len(config.attack_scenarios):
        raise PlanningError("resolved attack scenarios do not match the experiment configuration")
    for scenario_index, (resolved_scenario, configured_scenario) in enumerate(
        zip(resolved.attack_scenarios, config.attack_scenarios)
    ):
        location = f"resolved.attack_scenarios[{scenario_index}]"
        if resolved_scenario.id != configured_scenario.id:
            raise PlanningError(f"{location}.id does not match the configuration")
        if resolved_scenario.description != configured_scenario.description:
            raise PlanningError(f"{location}.description does not match the configuration")
        if len(resolved_scenario.steps) != len(configured_scenario.steps):
            raise PlanningError(f"{location} has an unexpected number of steps")
        for step_index, (resolved_step, configured_step) in enumerate(
            zip(resolved_scenario.steps, configured_scenario.steps)
        ):
            step_location = f"{location}.steps[{step_index}]"
            if resolved_step.index != step_index or resolved_step.name != configured_step.name:
                raise PlanningError(f"{step_location} does not match the configuration")
            if resolved_step.params != configured_step.params:
                raise PlanningError(f"{step_location}.params does not match the configuration")
            _validate_solution_binding(
                kind=SolutionKind.ATTACK,
                name=resolved_step.name,
                implementation=resolved_step.implementation,
                spec=resolved_step.spec,
                location=step_location,
            )

    if len(resolved.metrics) != len(config.metrics):
        raise PlanningError("resolved metric bindings do not match the experiment configuration")
    for index, (resolved_metric, configured_metric) in enumerate(zip(resolved.metrics, config.metrics)):
        location = f"resolved.metrics[{index}]"
        if (
            resolved_metric.id,
            resolved_metric.name,
            resolved_metric.checkpoint,
        ) != (
            configured_metric.id,
            configured_metric.name,
            configured_metric.checkpoint,
        ):
            raise PlanningError(f"{location} does not match the configuration")
        if dict(resolved_metric.inputs) != dict(configured_metric.inputs):
            raise PlanningError(f"{location}.inputs does not match the configuration")
        if resolved_metric.params != configured_metric.params:
            raise PlanningError(f"{location}.params does not match the configuration")
        _validate_solution_binding(
            kind=SolutionKind.METRIC,
            name=resolved_metric.name,
            implementation=resolved_metric.implementation,
            spec=resolved_metric.spec,
            location=location,
        )


def _validate_manifest_alignment(resolved: ResolvedExperiment, manifest: DatasetManifest) -> None:
    config = resolved.config
    configured_root = config.dataset.path.expanduser().resolve(strict=False)
    expected_extensions = tuple(sorted(item.lower() for item in config.dataset.extensions))
    expected_include = tuple(sorted(config.dataset.include))
    expected_exclude = tuple(sorted(config.dataset.exclude))

    mismatches = []
    if manifest.root != configured_root:
        mismatches.append(f"root is {manifest.root}, expected {configured_root}")
    if manifest.recursive != config.dataset.recursive:
        mismatches.append(f"recursive is {manifest.recursive}, expected {config.dataset.recursive}")
    if manifest.extensions != expected_extensions:
        mismatches.append(f"extensions are {manifest.extensions!r}, expected {expected_extensions!r}")
    if manifest.include != expected_include:
        mismatches.append(f"include patterns are {manifest.include!r}, expected {expected_include!r}")
    if manifest.exclude != expected_exclude:
        mismatches.append(f"exclude patterns are {manifest.exclude!r}, expected {expected_exclude!r}")
    if manifest.shuffled != config.dataset.shuffle:
        mismatches.append(f"shuffled is {manifest.shuffled}, expected {config.dataset.shuffle}")
    if manifest.seed != config.experiment.seed:
        mismatches.append(f"seed is {manifest.seed}, expected {config.experiment.seed}")
    if manifest.limit != config.dataset.limit:
        mismatches.append(f"limit is {manifest.limit!r}, expected {config.dataset.limit!r}")
    if mismatches:
        raise PlanningError("dataset manifest does not match the experiment configuration: " + "; ".join(mismatches))


@dataclass(frozen=True)
class ExperimentPlan:
    """A compact immutable plan whose variants and cases are generated lazily."""

    resolved: ResolvedExperiment
    manifest: DatasetManifest
    counts: PlanCounts
    fingerprint: str
    case_context_id: str
    schema_version: int = PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.resolved, ResolvedExperiment):
            raise TypeError("resolved must be a ResolvedExperiment")
        if not isinstance(self.manifest, DatasetManifest):
            raise TypeError("manifest must be a DatasetManifest")
        if not isinstance(self.counts, PlanCounts):
            raise TypeError("counts must be PlanCounts")
        if self.schema_version != PLAN_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {PLAN_SCHEMA_VERSION}")
        for field_name in ("fingerprint", "case_context_id"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field_name} must be a lowercase hexadecimal SHA-256 digest")

        expected_case_context_id = _case_context_id(self.resolved)
        if self.case_context_id != expected_case_context_id:
            raise ValueError("case_context_id does not match the resolved experiment")
        expected_fingerprint = _stable_digest(
            "experiment-plan",
            _plan_fingerprint_payload(self.resolved, self.manifest),
        )
        if self.fingerprint != expected_fingerprint:
            raise ValueError("fingerprint does not match the resolved experiment and dataset manifest")

        expected_case_count = (
            self.counts.sample_count
            * self.counts.embedding_variant_count
            * self.counts.attack_variant_count
            * self.counts.repeat_count
        )
        if self.counts.sample_count != len(self.manifest.samples):
            raise ValueError("counts.sample_count does not match the dataset manifest")
        if self.counts.repeat_count != self.resolved.config.experiment.repeats:
            raise ValueError("counts.repeat_count does not match the experiment configuration")
        if self.counts.case_count != expected_case_count:
            raise ValueError("counts.case_count is inconsistent with the plan dimensions")
        if self.counts.embedding_execution_count != self.counts.work_unit_count:
            raise ValueError("counts.embedding_execution_count must equal work_unit_count")
        if self.counts.extraction_execution_count != self.counts.case_count:
            raise ValueError("counts.extraction_execution_count must equal case_count")
        expected_metric_evaluations = {
            "after_embedding": (self.counts.work_unit_count * self.counts.metric_variant_counts["after_embedding"]),
            "after_attack": self.counts.case_count * self.counts.metric_variant_counts["after_attack"],
            "after_extraction": (self.counts.case_count * self.counts.metric_variant_counts["after_extraction"]),
        }
        if dict(self.counts.metric_evaluation_counts) != expected_metric_evaluations:
            raise ValueError("metric evaluation counts are inconsistent with plan cardinalities")

    def iter_embedding_variants(self) -> Iterator[EmbeddingVariant]:
        """Yield embedding variants in stable declaration and parameter order."""

        ordinal = 0
        for declaration in self.resolved.embeddings:
            solution = _solution_identity(SolutionKind.EMBEDDING, declaration.implementation, declaration.spec)
            for embedding_parameters, extraction_parameters in _iter_embedding_parameter_pairs(declaration):
                variant_id = _stable_digest(
                    "embedding-variant",
                    {
                        "declaration_id": declaration.id,
                        "solution": solution,
                        "embedding_parameters": embedding_parameters,
                        "extraction_parameters": extraction_parameters,
                    },
                )
                yield EmbeddingVariant(
                    variant_id=variant_id,
                    declaration_id=declaration.id,
                    solution_name=declaration.name,
                    ordinal=ordinal,
                    embedding_parameters=embedding_parameters,
                    extraction_parameters=extraction_parameters,
                    deterministic=declaration.spec.deterministic,
                    resource=declaration.spec.resource,
                )
                ordinal += 1

    def _iter_attack_step_variants(self, scenario: ResolvedAttackScenario, step: ResolvedAttackStep):
        solution = _solution_identity(SolutionKind.ATTACK, step.implementation, step.spec)
        for parameter_ordinal, parameters in enumerate(
            _iter_operation_parameters(step.params, step.spec.operations["attack"])
        ):
            variant_id = _stable_digest(
                "attack-step-variant",
                {
                    "scenario_id": scenario.id,
                    "step_index": step.index,
                    "solution": solution,
                    "parameters": parameters,
                },
            )
            yield AttackStepVariant(
                variant_id=variant_id,
                step_index=step.index,
                solution_name=step.name,
                parameter_ordinal=parameter_ordinal,
                parameters=parameters,
                deterministic=step.spec.deterministic,
                resource=step.spec.resource,
            )

    def _iter_scenario_variants(self, scenario: ResolvedAttackScenario) -> Iterator[AttackScenarioVariant]:
        if not scenario.steps:
            variant_id = _stable_digest(
                "attack-scenario-variant",
                {"scenario_id": scenario.id, "steps": []},
            )
            yield AttackScenarioVariant(
                variant_id=variant_id,
                scenario_id=scenario.id,
                description=scenario.description,
                ordinal=0,
                steps=(),
            )
            return

        def expand(step_index: int, selected: tuple[AttackStepVariant, ...]):
            if step_index == len(scenario.steps):
                yield selected
                return
            step = scenario.steps[step_index]
            for variant in self._iter_attack_step_variants(scenario, step):
                yield from expand(step_index + 1, selected + (variant,))

        for ordinal, steps in enumerate(expand(0, ())):
            variant_id = _stable_digest(
                "attack-scenario-variant",
                {
                    "scenario_id": scenario.id,
                    "steps": [step.variant_id for step in steps],
                },
            )
            yield AttackScenarioVariant(
                variant_id=variant_id,
                scenario_id=scenario.id,
                description=scenario.description,
                ordinal=ordinal,
                steps=steps,
            )

    def iter_attack_variants(self) -> Iterator[AttackScenarioVariant]:
        """Yield all clean and parameterised attack-chain variants lazily."""

        ordinal = 0
        for scenario in self.resolved.attack_scenarios:
            for variant in self._iter_scenario_variants(scenario):
                yield AttackScenarioVariant(
                    variant_id=variant.variant_id,
                    scenario_id=variant.scenario_id,
                    description=variant.description,
                    ordinal=ordinal,
                    steps=variant.steps,
                )
                ordinal += 1

    def iter_metric_variants(self, checkpoint: Optional[str] = None) -> Iterator[MetricVariant]:
        """Yield metric variants, optionally restricted to one checkpoint."""

        if checkpoint is not None and checkpoint not in _CHECKPOINTS:
            raise ValueError(f"unknown metric checkpoint {checkpoint!r}")
        ordinal = 0
        for metric in self.resolved.metrics:
            operation = metric.spec.operations["expertise"]
            solution = _solution_identity(SolutionKind.METRIC, metric.implementation, metric.spec)
            for parameters in _iter_operation_parameters(metric.params, operation):
                if checkpoint is None or metric.checkpoint == checkpoint:
                    variant_id = _stable_digest(
                        "metric-variant",
                        {
                            "binding_id": metric.id,
                            "solution": solution,
                            "checkpoint": metric.checkpoint,
                            "inputs": dict(metric.inputs),
                            "parameters": parameters,
                        },
                    )
                    yield MetricVariant(
                        variant_id=variant_id,
                        binding_id=metric.id,
                        solution_name=metric.name,
                        checkpoint=metric.checkpoint,
                        ordinal=ordinal,
                        inputs=metric.inputs,
                        parameters=parameters,
                        deterministic=metric.spec.deterministic,
                        resource=metric.spec.resource,
                    )
                ordinal += 1

    def _watermark_reference(
        self,
        sample: SampleReference,
        embedding: EmbeddingVariant,
        attack: Optional[AttackScenarioVariant],
        repeat_index: int,
    ) -> WatermarkReference:
        watermark = self.resolved.config.watermark
        if isinstance(watermark, FixedBitsWatermarkSpec):
            return WatermarkReference(
                watermark_id=_stable_digest(
                    "fixed-watermark",
                    {"bits": watermark.bits},
                ),
                kind="fixed_bits",
                scope="fixed_for_run",
                length=len(watermark.bits),
            )

        if not isinstance(watermark, RandomBitsWatermarkSpec):
            raise PlanningError(f"unsupported watermark specification {type(watermark).__name__}")
        seed_parts: list[Any] = [
            self.resolved.config.experiment.seed,
            watermark.length,
            watermark.scope,
        ]
        if watermark.scope in {"per_image", "per_case"}:
            seed_parts.append(sample.sample_id)
        if watermark.scope == "per_case":
            if attack is None:
                raise PlanningError("per_case watermark generation requires an attack variant")
            seed_parts.extend(
                [
                    embedding.variant_id,
                    attack.variant_id,
                    repeat_index,
                ]
            )
        seed = derive_stable_seed("watermark", *seed_parts)
        return WatermarkReference(
            watermark_id=_stable_digest(
                "random-watermark",
                {
                    "scope": watermark.scope,
                    "length": watermark.length,
                    "seed": seed,
                },
            ),
            kind="random_bits",
            scope=watermark.scope,
            length=watermark.length,
            seed=seed,
        )

    def _embedding_seed(
        self,
        sample: SampleReference,
        embedding: EmbeddingVariant,
        repeat_index: int,
        watermark: WatermarkReference,
    ) -> int:
        return derive_stable_seed(
            "embedding",
            self.resolved.config.experiment.seed,
            sample.sample_id,
            embedding.variant_id,
            repeat_index,
            watermark.watermark_id,
        )

    def _case_id(
        self,
        sample: SampleReference,
        embedding: EmbeddingVariant,
        attack: AttackScenarioVariant,
        repeat_index: int,
        watermark: WatermarkReference,
    ) -> str:
        return _stable_digest(
            "experiment-case",
            {
                "case_context_id": self.case_context_id,
                "sample_id": sample.sample_id,
                "embedding_variant_id": embedding.variant_id,
                "attack_variant_id": attack.variant_id,
                "repeat_index": repeat_index,
                "watermark_id": watermark.watermark_id,
            },
        )

    def _case_seed(
        self,
        sample: SampleReference,
        embedding: EmbeddingVariant,
        attack: AttackScenarioVariant,
        repeat_index: int,
        watermark: WatermarkReference,
    ) -> int:
        return derive_stable_seed(
            "case",
            self.resolved.config.experiment.seed,
            sample.sample_id,
            embedding.variant_id,
            attack.variant_id,
            repeat_index,
            watermark.watermark_id,
        )

    @property
    def _uses_per_case_watermarks(self) -> bool:
        watermark = self.resolved.config.watermark
        return isinstance(watermark, RandomBitsWatermarkSpec) and watermark.scope == "per_case"

    def _shared_work_unit_id(
        self,
        sample: SampleReference,
        embedding: EmbeddingVariant,
        repeat_index: int,
        watermark: WatermarkReference,
    ) -> str:
        return _stable_digest(
            "work-unit",
            {
                "case_context_id": self.case_context_id,
                "sample_id": sample.sample_id,
                "embedding_variant_id": embedding.variant_id,
                "repeat_index": repeat_index,
                "watermark_id": watermark.watermark_id,
            },
        )

    def _make_case(
        self,
        *,
        ordinal: int,
        work_unit_id: str,
        sample: SampleReference,
        embedding: EmbeddingVariant,
        attack: AttackScenarioVariant,
        repeat_index: int,
        embedding_seed: int,
        watermark: WatermarkReference,
    ) -> ExperimentCase:
        return ExperimentCase(
            case_id=self._case_id(
                sample,
                embedding,
                attack,
                repeat_index,
                watermark,
            ),
            work_unit_id=work_unit_id,
            ordinal=ordinal,
            sample=sample,
            embedding=embedding,
            attack=attack,
            repeat_index=repeat_index,
            embedding_seed=embedding_seed,
            case_seed=self._case_seed(
                sample,
                embedding,
                attack,
                repeat_index,
                watermark,
            ),
            watermark=watermark,
        )

    def iter_work_units(self) -> Iterator[WorkUnit]:
        """Yield coarse units while materialising only the current unit's cases."""

        work_unit_ordinal = 0
        case_ordinal = 0
        for sample in self.manifest.samples:
            for embedding in self.iter_embedding_variants():
                for repeat_index in range(self.resolved.config.experiment.repeats):
                    if self._uses_per_case_watermarks:
                        for attack in self.iter_attack_variants():
                            watermark = self._watermark_reference(
                                sample,
                                embedding,
                                attack,
                                repeat_index,
                            )
                            case_id = self._case_id(
                                sample,
                                embedding,
                                attack,
                                repeat_index,
                                watermark,
                            )
                            work_unit_id = _stable_digest("work-unit", {"case_id": case_id})
                            embedding_seed = self._embedding_seed(
                                sample,
                                embedding,
                                repeat_index,
                                watermark,
                            )
                            case = self._make_case(
                                ordinal=case_ordinal,
                                work_unit_id=work_unit_id,
                                sample=sample,
                                embedding=embedding,
                                attack=attack,
                                repeat_index=repeat_index,
                                embedding_seed=embedding_seed,
                                watermark=watermark,
                            )
                            yield WorkUnit(
                                work_unit_id=work_unit_id,
                                ordinal=work_unit_ordinal,
                                sample=sample,
                                embedding=embedding,
                                repeat_index=repeat_index,
                                embedding_seed=embedding_seed,
                                watermark=watermark,
                                cases=(case,),
                            )
                            case_ordinal += 1
                            work_unit_ordinal += 1
                        continue

                    watermark = self._watermark_reference(
                        sample,
                        embedding,
                        None,
                        repeat_index,
                    )
                    work_unit_id = self._shared_work_unit_id(
                        sample,
                        embedding,
                        repeat_index,
                        watermark,
                    )
                    embedding_seed = self._embedding_seed(
                        sample,
                        embedding,
                        repeat_index,
                        watermark,
                    )
                    cases = []
                    for attack in self.iter_attack_variants():
                        cases.append(
                            self._make_case(
                                ordinal=case_ordinal,
                                work_unit_id=work_unit_id,
                                sample=sample,
                                embedding=embedding,
                                attack=attack,
                                repeat_index=repeat_index,
                                embedding_seed=embedding_seed,
                                watermark=watermark,
                            )
                        )
                        case_ordinal += 1
                    yield WorkUnit(
                        work_unit_id=work_unit_id,
                        ordinal=work_unit_ordinal,
                        sample=sample,
                        embedding=embedding,
                        repeat_index=repeat_index,
                        embedding_seed=embedding_seed,
                        watermark=watermark,
                        cases=tuple(cases),
                    )
                    work_unit_ordinal += 1

    def iter_cases(self) -> Iterator[ExperimentCase]:
        """Yield cases one at a time without materialising a work-unit batch."""

        case_ordinal = 0
        for sample in self.manifest.samples:
            for embedding in self.iter_embedding_variants():
                for repeat_index in range(self.resolved.config.experiment.repeats):
                    shared_watermark = None
                    shared_work_unit_id = None
                    if not self._uses_per_case_watermarks:
                        shared_watermark = self._watermark_reference(
                            sample,
                            embedding,
                            None,
                            repeat_index,
                        )
                        shared_work_unit_id = self._shared_work_unit_id(
                            sample,
                            embedding,
                            repeat_index,
                            shared_watermark,
                        )

                    for attack in self.iter_attack_variants():
                        watermark = shared_watermark or self._watermark_reference(
                            sample,
                            embedding,
                            attack,
                            repeat_index,
                        )
                        if shared_work_unit_id is None:
                            case_id = self._case_id(
                                sample,
                                embedding,
                                attack,
                                repeat_index,
                                watermark,
                            )
                            work_unit_id = _stable_digest("work-unit", {"case_id": case_id})
                        else:
                            work_unit_id = shared_work_unit_id
                        embedding_seed = self._embedding_seed(
                            sample,
                            embedding,
                            repeat_index,
                            watermark,
                        )
                        yield self._make_case(
                            ordinal=case_ordinal,
                            work_unit_id=work_unit_id,
                            sample=sample,
                            embedding=embedding,
                            attack=attack,
                            repeat_index=repeat_index,
                            embedding_seed=embedding_seed,
                            watermark=watermark,
                        )
                        case_ordinal += 1

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-compatible description without enumerating cases."""

        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "case_context_id": self.case_context_id,
            "catalog_fingerprint": self.resolved.catalog_fingerprint,
            "manifest_fingerprint": self.manifest.fingerprint,
            "experiment_name": self.resolved.config.experiment.name,
            "counts": self.counts.to_dict(),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def summary(self) -> str:
        """Return a concise human-readable cardinality summary."""

        counts = self.counts
        lines = [
            f"Experiment: {self.resolved.config.experiment.name}",
            f"Plan fingerprint: {self.fingerprint}",
            f"Dataset samples: {counts.sample_count}",
            f"Embedding variants: {counts.embedding_variant_count}",
            f"Attack variants: {counts.attack_variant_count}",
            f"Repeats: {counts.repeat_count}",
            f"Experiment cases: {counts.case_count}",
            f"Work units: {counts.work_unit_count}",
            f"Metric evaluations: {counts.metric_evaluation_count}",
        ]
        return "\n".join(lines)


class ExperimentPlanner:
    """Build deterministic plans from resolved experiments and dataset manifests."""

    __slots__ = ("manifest", "resolved")

    def __init__(self, resolved: ResolvedExperiment, manifest: DatasetManifest) -> None:
        if not isinstance(resolved, ResolvedExperiment):
            raise TypeError("resolved must be a ResolvedExperiment")
        if not isinstance(manifest, DatasetManifest):
            raise TypeError("manifest must be a DatasetManifest")
        self.resolved = resolved
        self.manifest = manifest

    def _counts(self) -> PlanCounts:
        sample_count = len(self.manifest.samples)
        embedding_variant_count = sum(_embedding_variant_count(item) for item in self.resolved.embeddings)
        scenario_counts = tuple(_attack_scenario_variant_count(scenario) for scenario in self.resolved.attack_scenarios)
        attack_variant_count = sum(scenario_counts)
        repeat_count = self.resolved.config.experiment.repeats
        case_count = sample_count * embedding_variant_count * attack_variant_count * repeat_count

        per_case_watermark = (
            isinstance(self.resolved.config.watermark, RandomBitsWatermarkSpec)
            and self.resolved.config.watermark.scope == "per_case"
        )

        work_unit_count = case_count if per_case_watermark else sample_count * embedding_variant_count * repeat_count

        attack_variants_with_steps = sum(
            variant_count * len(scenario.steps)
            for scenario, variant_count in zip(self.resolved.attack_scenarios, scenario_counts)
        )
        attack_execution_count = sample_count * embedding_variant_count * repeat_count * attack_variants_with_steps

        metric_variant_counts = _metric_variant_counts(self.resolved)
        metric_evaluation_counts = {
            "after_embedding": work_unit_count * metric_variant_counts["after_embedding"],
            "after_attack": case_count * metric_variant_counts["after_attack"],
            "after_extraction": case_count * metric_variant_counts["after_extraction"],
        }
        return PlanCounts(
            sample_count=sample_count,
            embedding_variant_count=embedding_variant_count,
            attack_variant_count=attack_variant_count,
            repeat_count=repeat_count,
            case_count=case_count,
            work_unit_count=work_unit_count,
            embedding_execution_count=work_unit_count,
            attack_execution_count=attack_execution_count,
            extraction_execution_count=case_count,
            metric_variant_counts=metric_variant_counts,
            metric_evaluation_counts=metric_evaluation_counts,
        )

    def build(self) -> ExperimentPlan:
        """Validate inputs, calculate cardinalities and return a lazy plan."""

        _validate_resolved_alignment(self.resolved)
        _validate_manifest_alignment(self.resolved, self.manifest)
        counts = self._counts()
        experiment = self.resolved.config.experiment
        if counts.case_count > experiment.max_cases and not experiment.allow_large_plan:
            raise PlanTooLargeError(
                case_count=counts.case_count,
                max_cases=experiment.max_cases,
            )

        fingerprint = _stable_digest(
            "experiment-plan",
            _plan_fingerprint_payload(self.resolved, self.manifest),
        )
        return ExperimentPlan(
            resolved=self.resolved,
            manifest=self.manifest,
            counts=counts,
            fingerprint=fingerprint,
            case_context_id=_case_context_id(self.resolved),
        )


def build_experiment_plan(
    resolved: ResolvedExperiment,
    manifest: DatasetManifest,
) -> ExperimentPlan:
    """Convenience wrapper around :class:`ExperimentPlanner`."""

    return ExperimentPlanner(resolved, manifest).build()


__all__ = [
    "ExperimentPlan",
    "ExperimentPlanner",
    "build_experiment_plan",
    "derive_stable_seed",
]
