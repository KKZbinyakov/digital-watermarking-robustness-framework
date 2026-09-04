# ruff: noqa: UP007, UP045
"""Discovery and registry-dependent validation for experiment configurations."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import Any, Optional

from dwarf.core.attack_orchestrator.attack_core import Attack_Core
from dwarf.core.embedding_orchestrator.embedding_core import Embedding_Core
from dwarf.core.expertise_orchestrator.expertise_core import Expertise_Core
from dwarf.pipeline.config import ExperimentConfig, ParameterSpace
from dwarf.pipeline.exceptions import (
    SemanticValidationError,
    SemanticValidationIssue,
    SolutionConflictError,
    SolutionDiscoveryError,
)
from dwarf.pipeline.resolved import (
    ResolvedAttackScenario,
    ResolvedAttackStep,
    ResolvedEmbedding,
    ResolvedExperiment,
    ResolvedMetric,
)
from dwarf.pipeline.solution_spec import (
    UNSET,
    ArtifactInputSpec,
    ArtifactKind,
    DataContract,
    FrozenDict,
    OperationSpec,
    ParameterKind,
    ParameterLinkSpec,
    ParameterSpec,
    SolutionKind,
    SolutionSpec,
)

_REGISTRY_GETTERS = {
    SolutionKind.ATTACK: Attack_Core.get_registered_attacks,
    SolutionKind.EMBEDDING: Embedding_Core.get_registered_embeddings,
    SolutionKind.METRIC: Expertise_Core.get_registered_expertises,
}
_REQUIRED_OPERATIONS = {
    SolutionKind.ATTACK: ("attack",),
    SolutionKind.EMBEDDING: ("embedding", "extraction"),
    SolutionKind.METRIC: ("expertise",),
}

_DCT_SPEC = SolutionSpec(
    name="DCT",
    kind=SolutionKind.EMBEDDING,
    operations={
        "embedding": OperationSpec(
            entry_point="embedding",
            parameters={
                "margin": ParameterSpec(
                    ParameterKind.NUMBER,
                    default=150.0,
                    minimum=0.0,
                ),
                "threshold": ParameterSpec(
                    ParameterKind.NUMBER,
                    default=25.0,
                    minimum=0.0,
                ),
            },
            runtime_arguments=("input_image", "watermark_bits"),
        ),
        "extraction": OperationSpec(
            entry_point="extraction",
            parameters={
                "threshold": ParameterSpec(
                    ParameterKind.NUMBER,
                    default=25.0,
                    minimum=0.0,
                ),
            },
            runtime_arguments=("input_image", "num_bits"),
        ),
    },
    contracts={
        "embedding_input": DataContract.LUMA_FLOAT64,
        "embedding_output": DataContract.LUMA_FLOAT64,
        "original_watermark": DataContract.BINARY_BITS,
        "extracted_watermark": DataContract.TERNARY_BITS,
    },
    parameter_links=(
        ParameterLinkSpec(
            source_operation="embedding",
            source_parameter="threshold",
            target_operation="extraction",
            target_parameter="threshold",
        ),
    ),
)

_JPEG_SPEC = SolutionSpec(
    name="Jpeg",
    kind=SolutionKind.ATTACK,
    operations={
        "attack": OperationSpec(
            entry_point="attack",
            parameters={
                "quality": ParameterSpec(
                    ParameterKind.INTEGER,
                    default=75,
                    minimum=0,
                    maximum=100,
                )
            },
            runtime_arguments=("input_image",),
        )
    },
    contracts={
        "input_image": DataContract.RGB_UINT8,
        "output_image": DataContract.RGB_UINT8,
    },
)

_IMAGE_ARTIFACTS = ("original_image", "embedded_image", "attacked_image")
_PSNR_SPEC = SolutionSpec(
    name="PSNR",
    kind=SolutionKind.METRIC,
    operations={
        "expertise": OperationSpec(
            entry_point="expertise",
            artifact_inputs={
                "original_image": ArtifactInputSpec(_IMAGE_ARTIFACTS, ArtifactKind.IMAGE),
                "distorted_image": ArtifactInputSpec(_IMAGE_ARTIFACTS, ArtifactKind.IMAGE),
            },
            checkpoints=("after_embedding", "after_attack", "after_extraction"),
        )
    },
    contracts={"output": DataContract.SCALAR_NUMBER},
)

_BER_SPEC = SolutionSpec(
    name="BER",
    kind=SolutionKind.METRIC,
    operations={
        "expertise": OperationSpec(
            entry_point="expertise",
            parameters={
                "allow_length_mismatch": ParameterSpec(
                    ParameterKind.BOOLEAN,
                    default=False,
                )
            },
            artifact_inputs={
                "original_bits": ArtifactInputSpec(
                    ("original_watermark",),
                    ArtifactKind.WATERMARK,
                ),
                "extracted_bits": ArtifactInputSpec(
                    ("extracted_watermark",),
                    ArtifactKind.WATERMARK,
                ),
            },
            checkpoints=("after_extraction",),
        )
    },
    contracts={"output": DataContract.SCALAR_NUMBER},
)

_BUILTIN_SPECS = (_DCT_SPEC, _JPEG_SPEC, _PSNR_SPEC, _BER_SPEC)


def _snapshot_registries() -> dict[SolutionKind, dict[str, type]]:
    return {
        kind: {name: implementation for name, implementation in getter().items() if not name.startswith("Ready_")}
        for kind, getter in _REGISTRY_GETTERS.items()
    }


def _restore_registries(snapshot: Mapping[SolutionKind, Mapping[str, type]]) -> None:
    for kind, getter in _REGISTRY_GETTERS.items():
        registry = getter()
        ready_classes = {name: value for name, value in registry.items() if name.startswith("Ready_")}
        registry.clear()
        registry.update(ready_classes)
        registry.update(snapshot[kind])


def _registry_replacements(
    before: Mapping[SolutionKind, Mapping[str, type]],
    after: Mapping[SolutionKind, Mapping[str, type]],
) -> list[tuple[SolutionKind, str]]:
    replacements = []
    for kind in SolutionKind:
        for name, implementation in before[kind].items():
            if name in after[kind] and after[kind][name] is not implementation:
                replacements.append((kind, name))
    return replacements


def _normalise_specs(value: Any, owner: str) -> tuple[SolutionSpec, ...]:
    if value is None:
        return ()
    if isinstance(value, SolutionSpec):
        return (value,)
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise SolutionDiscoveryError(f"{owner} must contain SolutionSpec objects")
    specs = tuple(value)
    if any(not isinstance(item, SolutionSpec) for item in specs):
        raise SolutionDiscoveryError(f"{owner} must contain only SolutionSpec objects")
    return specs


def _add_spec(
    target: dict[tuple[SolutionKind, str], SolutionSpec],
    spec: SolutionSpec,
    *,
    owner: str,
) -> None:
    key = (spec.kind, spec.name)
    previous = target.get(key)
    if previous is not None and previous != spec:
        raise SolutionConflictError(
            f"conflicting pipeline metadata for {spec.kind.value} solution {spec.name!r}; "
            f"second declaration came from {owner}"
        )
    target[key] = spec


def _validate_spec_for_implementation(
    spec: SolutionSpec,
    implementation: type,
    kind: SolutionKind,
    owner: str,
) -> None:
    if spec.name != implementation.__name__:
        raise SolutionDiscoveryError(
            f"{owner} declares solution name {spec.name!r}, but the registered class is {implementation.__name__!r}"
        )
    if spec.kind is not kind:
        raise SolutionDiscoveryError(
            f"{owner} declares kind {spec.kind.value!r}, but {implementation.__name__!r} is registered as {kind.value}"
        )
    missing = [name for name in _REQUIRED_OPERATIONS[kind] if name not in spec.operations]
    if missing:
        raise SolutionDiscoveryError(f"{owner} does not describe required operations: {', '.join(missing)}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    return value


def _parameter_locations(space: ParameterSpace, parameter_name: str, prefix: str):
    if parameter_name in space.fixed:
        yield f"{prefix}.fixed.{parameter_name}", space.fixed[parameter_name]
    if parameter_name in space.grid:
        for index, value in enumerate(space.grid[parameter_name]):
            yield f"{prefix}.grid.{parameter_name}[{index}]", value
    if space.variants and parameter_name in space.variants[0]:
        for index, variant in enumerate(space.variants):
            yield f"{prefix}.variants[{index}].{parameter_name}", variant[parameter_name]


def _declared_parameter_names(space: ParameterSpace) -> set[str]:
    result = set(space.fixed) | set(space.grid)
    if space.variants:
        result.update(space.variants[0])
    return result


def _canonical_parameter_value(value: Any, parameter: ParameterSpec) -> str:
    plain = _plain_value(value)
    if parameter.kind is ParameterKind.NUMBER and isinstance(plain, (int, float)) and not isinstance(plain, bool):
        plain = float(plain)
    elif parameter.kind is ParameterKind.INTEGER and isinstance(plain, int) and not isinstance(plain, bool):
        plain = int(plain)
    return _canonical_json(plain)


def _parameter_axis(space: ParameterSpace, name: str, parameter: ParameterSpec):
    if name in space.fixed:
        return "scalar", _canonical_parameter_value(space.fixed[name], parameter)
    if name in space.grid:
        values = tuple(_canonical_parameter_value(item, parameter) for item in space.grid[name])
        if len(values) == 1:
            return "scalar", values[0]
        return "axis", values
    if space.variants and name in space.variants[0]:
        values = tuple(_canonical_parameter_value(item[name], parameter) for item in space.variants)
        if len(set(values)) == 1:
            return "scalar", values[0]
        return "variants", values
    if parameter.default is not UNSET:
        return "scalar", _canonical_parameter_value(parameter.default, parameter)
    return "missing", None


def _entry_point_issue(
    implementation: type,
    operation: OperationSpec,
    path: str,
) -> Optional[SemanticValidationIssue]:
    entry_point = getattr(implementation, operation.entry_point, None)
    if not callable(entry_point):
        return SemanticValidationIssue(
            path,
            "missing_entry_point",
            f"registered class {implementation.__module__}.{implementation.__qualname__} "
            f"does not provide callable {operation.entry_point}()",
        )
    try:
        signature = inspect.signature(entry_point)
    except (TypeError, ValueError):
        return None
    if not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return SemanticValidationIssue(
            path,
            "invalid_entry_point_signature",
            f"{operation.entry_point}() must accept arbitrary keyword arguments through **kwargs",
        )
    return None


class SolutionCatalog:
    """Immutable snapshot of registered classes and their pipeline metadata."""

    def __init__(
        self,
        registries: Mapping[SolutionKind, Mapping[str, type]],
        specs: Mapping[tuple[SolutionKind, str], SolutionSpec],
        *,
        plugins: Iterable[str] = (),
    ) -> None:
        normalised_registries = {}
        for kind in SolutionKind:
            entries = {}
            for name, implementation in registries.get(kind, {}).items():
                if name.startswith("Ready_"):
                    continue
                if not isinstance(name, str) or not isinstance(implementation, type):
                    raise SolutionDiscoveryError(f"invalid {kind.value} registry entry {name!r}")
                entries[name] = implementation
            normalised_registries[kind] = FrozenDict(entries)

        normalised_specs = {}
        for key, spec in specs.items():
            kind, name = key
            kind = SolutionKind(kind)
            if not isinstance(spec, SolutionSpec):
                raise SolutionDiscoveryError(f"metadata for {kind.value} solution {name!r} is not a SolutionSpec")
            if spec.kind is not kind or spec.name != name:
                raise SolutionDiscoveryError(f"metadata key {(kind.value, name)!r} does not match its SolutionSpec")
            normalised_specs[(kind, name)] = spec

        plugin_tuple = tuple(plugins)
        if len(plugin_tuple) != len(set(plugin_tuple)):
            raise SolutionDiscoveryError("plugins must not contain duplicate module paths")

        self._registries = FrozenDict(normalised_registries)
        self._specs = FrozenDict(normalised_specs)
        self._plugins = plugin_tuple
        self._fingerprint = self._build_fingerprint()

    @classmethod
    def discover(cls, *, plugins: Iterable[str] = ()) -> SolutionCatalog:
        """Import built-ins and plugins, then return a validated registry snapshot."""

        plugin_tuple = tuple(plugins)
        try:
            importlib.import_module("dwarf.ready_solutions")
        except Exception as error:
            raise SolutionDiscoveryError(f"could not import dwarf.ready_solutions: {error}") from error

        baseline = _snapshot_registries()
        modules: list[ModuleType] = []
        for module_name in plugin_tuple:
            before = _snapshot_registries()
            try:
                module = importlib.import_module(module_name)
            except Exception as error:
                _restore_registries(baseline)
                raise SolutionDiscoveryError(f"could not import plugin {module_name!r}: {error}") from error
            after = _snapshot_registries()
            replacements = _registry_replacements(before, after)
            if replacements:
                _restore_registries(baseline)
                details = ", ".join(f"{kind.value} {name!r}" for kind, name in replacements)
                raise SolutionConflictError(f"plugin {module_name!r} replaced registered solutions: {details}")
            modules.append(module)

        registries = _snapshot_registries()
        specs: dict[tuple[SolutionKind, str], SolutionSpec] = {}
        for spec in _BUILTIN_SPECS:
            if spec.name in registries[spec.kind]:
                _add_spec(specs, spec, owner="DWARF built-in metadata")

        try:
            for kind, entries in registries.items():
                for implementation in entries.values():
                    class_spec = implementation.__dict__.get("PIPELINE_SPEC")
                    if class_spec is None:
                        continue
                    for spec in _normalise_specs(
                        class_spec,
                        f"{implementation.__module__}.{implementation.__qualname__}.PIPELINE_SPEC",
                    ):
                        _validate_spec_for_implementation(
                            spec,
                            implementation,
                            kind,
                            f"{implementation.__module__}.{implementation.__qualname__}.PIPELINE_SPEC",
                        )
                        _add_spec(
                            specs,
                            spec,
                            owner=f"{implementation.__module__}.{implementation.__qualname__}.PIPELINE_SPEC",
                        )

            for module in modules:
                module_specs = _normalise_specs(
                    getattr(module, "DWARF_PIPELINE_SPECS", None),
                    f"{module.__name__}.DWARF_PIPELINE_SPECS",
                )
                for spec in module_specs:
                    implementation = registries[spec.kind].get(spec.name)
                    if implementation is None:
                        raise SolutionDiscoveryError(
                            f"{module.__name__}.DWARF_PIPELINE_SPECS describes {spec.kind.value} "
                            f"solution {spec.name!r}, but no class with that name is registered"
                        )
                    _validate_spec_for_implementation(
                        spec,
                        implementation,
                        spec.kind,
                        f"{module.__name__}.DWARF_PIPELINE_SPECS",
                    )
                    _add_spec(specs, spec, owner=f"{module.__name__}.DWARF_PIPELINE_SPECS")
        except Exception:
            _restore_registries(baseline)
            raise

        return cls(registries, specs, plugins=plugin_tuple)

    @classmethod
    def from_registries(
        cls,
        *,
        attacks: Optional[Mapping[str, type]] = None,
        embeddings: Optional[Mapping[str, type]] = None,
        metrics: Optional[Mapping[str, type]] = None,
        specs: Iterable[SolutionSpec] = (),
        plugins: Iterable[str] = (),
    ) -> SolutionCatalog:
        """Build a catalog from explicit snapshots, primarily for tests and tools."""

        registries = {
            SolutionKind.ATTACK: dict(attacks or {}),
            SolutionKind.EMBEDDING: dict(embeddings or {}),
            SolutionKind.METRIC: dict(metrics or {}),
        }
        indexed_specs = {}
        for spec in specs:
            _add_spec(indexed_specs, spec, owner="explicit catalog metadata")
        for kind, entries in registries.items():
            for implementation in entries.values():
                class_spec = implementation.__dict__.get("PIPELINE_SPEC")
                if class_spec is None:
                    continue
                for spec in _normalise_specs(class_spec, f"{implementation.__name__}.PIPELINE_SPEC"):
                    _validate_spec_for_implementation(
                        spec,
                        implementation,
                        kind,
                        f"{implementation.__name__}.PIPELINE_SPEC",
                    )
                    _add_spec(indexed_specs, spec, owner=f"{implementation.__name__}.PIPELINE_SPEC")
        return cls(registries, indexed_specs, plugins=plugins)

    @property
    def plugins(self) -> tuple[str, ...]:
        return self._plugins

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def registered(self, kind: SolutionKind) -> tuple[str, ...]:
        return tuple(sorted(self._registries[SolutionKind(kind)]))

    def available(self, kind: SolutionKind) -> tuple[str, ...]:
        kind = SolutionKind(kind)
        return tuple(sorted(name for name in self._registries[kind] if (kind, name) in self._specs))

    def implementation(self, kind: SolutionKind, name: str) -> type:
        return self._registries[SolutionKind(kind)][name]

    def spec(self, kind: SolutionKind, name: str) -> SolutionSpec:
        return self._specs[(SolutionKind(kind), name)]

    def _build_fingerprint(self) -> str:
        records = []
        for kind in SolutionKind:
            for name in sorted(self._registries[kind]):
                key = (kind, name)
                if key not in self._specs:
                    continue
                implementation = self._registries[kind][name]
                records.append(
                    {
                        "kind": kind.value,
                        "name": name,
                        "implementation": f"{implementation.__module__}.{implementation.__qualname__}",
                        "spec": self._specs[key].to_dict(),
                    }
                )
        payload = {"plugins": list(self._plugins), "solutions": records}
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def _resolve_class(
        self,
        kind: SolutionKind,
        name: str,
        path: str,
        issues: list[SemanticValidationIssue],
    ) -> Optional[tuple[type, SolutionSpec]]:
        implementation = self._registries[kind].get(name)
        if implementation is None:
            other_kinds = [other.value for other in SolutionKind if name in self._registries[other]]
            if other_kinds:
                issues.append(
                    SemanticValidationIssue(
                        path,
                        "wrong_solution_kind",
                        f"solution {name!r} is registered as {', '.join(other_kinds)}, not {kind.value}",
                    )
                )
            else:
                issues.append(
                    SemanticValidationIssue(
                        path,
                        "unknown_solution",
                        f"{kind.value} solution {name!r} is not registered",
                    )
                )
            return None

        spec = self._specs.get((kind, name))
        if spec is None:
            issues.append(
                SemanticValidationIssue(
                    path,
                    "unsupported_solution",
                    f"{kind.value} solution {name!r} is registered but does not declare pipeline metadata",
                )
            )
            return None

        for operation_name in _REQUIRED_OPERATIONS[kind]:
            operation = spec.operations.get(operation_name)
            if operation is None:
                issues.append(
                    SemanticValidationIssue(
                        path,
                        "missing_entry_point",
                        f"pipeline metadata does not describe required operation {operation_name!r}",
                    )
                )
                continue
            issue = _entry_point_issue(implementation, operation, path)
            if issue is not None:
                issues.append(issue)
        return implementation, spec

    @staticmethod
    def _validate_parameter_space(
        space: ParameterSpace,
        operation: OperationSpec,
        path: str,
        issues: list[SemanticValidationIssue],
    ) -> None:
        declared = _declared_parameter_names(space)
        for name in sorted(declared - set(operation.parameters)):
            first_location = next(_parameter_locations(space, name, f"{path}.params"))[0]
            issues.append(
                SemanticValidationIssue(
                    first_location,
                    "unknown_parameter",
                    f"operation {operation.entry_point!r} does not accept user parameter {name!r}",
                )
            )

        for name, parameter in operation.parameters.items():
            if parameter.required and name not in declared:
                issues.append(
                    SemanticValidationIssue(
                        f"{path}.params",
                        "missing_parameter",
                        f"required parameter {name!r} is not configured",
                    )
                )
                continue
            for location, value in _parameter_locations(space, name, f"{path}.params"):
                message = parameter.validation_error(value)
                if message is not None:
                    issues.append(
                        SemanticValidationIssue(
                            location,
                            "invalid_parameter",
                            f"parameter {name!r} {message}",
                        )
                    )

    @staticmethod
    def _validate_parameter_links(
        embedding_index: int,
        embedding_spec,
        solution_spec: SolutionSpec,
        issues: list[SemanticValidationIssue],
    ) -> None:
        operation_spaces = {
            "embedding": embedding_spec.embedding.params,
            "extraction": embedding_spec.extraction.params,
        }
        for link in solution_spec.parameter_links:
            source_parameter = solution_spec.operations[link.source_operation].parameters[link.source_parameter]
            target_parameter = solution_spec.operations[link.target_operation].parameters[link.target_parameter]
            source = _parameter_axis(
                operation_spaces[link.source_operation],
                link.source_parameter,
                source_parameter,
            )
            target = _parameter_axis(
                operation_spaces[link.target_operation],
                link.target_parameter,
                target_parameter,
            )
            if source != target:
                issues.append(
                    SemanticValidationIssue(
                        f"embeddings[{embedding_index}].{link.target_operation}.params",
                        "linked_parameter_mismatch",
                        f"{link.source_operation}.{link.source_parameter} and "
                        f"{link.target_operation}.{link.target_parameter} must declare "
                        "the same value or variation axis",
                    )
                )

    @staticmethod
    def _validate_metric(
        metric,
        operation: OperationSpec,
        path: str,
        issues: list[SemanticValidationIssue],
    ) -> None:
        if operation.checkpoints and metric.checkpoint not in operation.checkpoints:
            issues.append(
                SemanticValidationIssue(
                    f"{path}.checkpoint",
                    "unsupported_checkpoint",
                    f"metric supports checkpoints {operation.checkpoints!r}, not {metric.checkpoint!r}",
                )
            )

        configured_inputs = set(metric.inputs)
        known_inputs = set(operation.artifact_inputs)
        for name in sorted(configured_inputs - known_inputs):
            issues.append(
                SemanticValidationIssue(
                    f"{path}.inputs.{name}",
                    "unknown_metric_input",
                    f"metric does not accept artifact input {name!r}",
                )
            )
        for name, artifact_input in operation.artifact_inputs.items():
            if artifact_input.required and name not in configured_inputs:
                issues.append(
                    SemanticValidationIssue(
                        f"{path}.inputs",
                        "missing_metric_input",
                        f"required metric input {name!r} is not bound",
                    )
                )
            elif name in configured_inputs and metric.inputs[name] not in artifact_input.artifacts:
                issues.append(
                    SemanticValidationIssue(
                        f"{path}.inputs.{name}",
                        "incompatible_artifact",
                        f"input {name!r} accepts {artifact_input.artifacts!r}, not {metric.inputs[name]!r}",
                    )
                )

    def resolve(self, config: ExperimentConfig) -> ResolvedExperiment:
        """Resolve every configured name and aggregate all semantic failures."""

        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig")

        issues: list[SemanticValidationIssue] = []
        loaded_plugins = set(self._plugins)
        for index, plugin in enumerate(config.plugins):
            if plugin not in loaded_plugins:
                issues.append(
                    SemanticValidationIssue(
                        f"plugins[{index}]",
                        "plugin_not_loaded",
                        f"plugin {plugin!r} is declared by the configuration but is not loaded in this catalog",
                    )
                )

        resolved_embeddings = []
        for index, embedding in enumerate(config.embeddings):
            path = f"embeddings[{index}]"
            resolved_class = self._resolve_class(
                SolutionKind.EMBEDDING,
                embedding.name,
                f"{path}.name",
                issues,
            )
            if resolved_class is None:
                continue
            implementation, spec = resolved_class
            self._validate_parameter_space(
                embedding.embedding.params,
                spec.operations["embedding"],
                f"{path}.embedding",
                issues,
            )
            self._validate_parameter_space(
                embedding.extraction.params,
                spec.operations["extraction"],
                f"{path}.extraction",
                issues,
            )
            self._validate_parameter_links(index, embedding, spec, issues)
            resolved_embeddings.append(
                ResolvedEmbedding(
                    id=embedding.id,
                    name=embedding.name,
                    implementation=implementation,
                    spec=spec,
                    embedding=embedding.embedding,
                    extraction=embedding.extraction,
                )
            )

        resolved_scenarios = []
        for scenario_index, scenario in enumerate(config.attack_scenarios):
            resolved_steps = []
            for step_index, step in enumerate(scenario.steps):
                path = f"attack_scenarios[{scenario_index}].steps[{step_index}]"
                resolved_class = self._resolve_class(
                    SolutionKind.ATTACK,
                    step.name,
                    f"{path}.name",
                    issues,
                )
                if resolved_class is None:
                    continue
                implementation, spec = resolved_class
                self._validate_parameter_space(
                    step.params,
                    spec.operations["attack"],
                    path,
                    issues,
                )
                resolved_steps.append(
                    ResolvedAttackStep(
                        index=step_index,
                        name=step.name,
                        implementation=implementation,
                        spec=spec,
                        params=step.params,
                    )
                )
            resolved_scenarios.append(
                ResolvedAttackScenario(
                    id=scenario.id,
                    description=scenario.description,
                    steps=tuple(resolved_steps),
                )
            )

        resolved_metrics = []
        for index, metric in enumerate(config.metrics):
            path = f"metrics[{index}]"
            resolved_class = self._resolve_class(
                SolutionKind.METRIC,
                metric.name,
                f"{path}.name",
                issues,
            )
            if resolved_class is None:
                continue
            implementation, spec = resolved_class
            operation = spec.operations["expertise"]
            self._validate_parameter_space(metric.params, operation, path, issues)
            self._validate_metric(metric, operation, path, issues)
            resolved_metrics.append(
                ResolvedMetric(
                    id=metric.id,
                    name=metric.name,
                    implementation=implementation,
                    spec=spec,
                    checkpoint=metric.checkpoint,
                    inputs=metric.inputs,
                    params=metric.params,
                )
            )

        if issues:
            raise SemanticValidationError(issues)
        return ResolvedExperiment(
            config=config,
            embeddings=tuple(resolved_embeddings),
            attack_scenarios=tuple(resolved_scenarios),
            metrics=tuple(resolved_metrics),
            plugins=self._plugins,
            catalog_fingerprint=self._fingerprint,
        )


__all__ = ["SolutionCatalog"]
