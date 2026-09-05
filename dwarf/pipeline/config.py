# ruff: noqa: UP007, UP045
"""Strict, versioned configuration models for DWARF experiments.

The models in this module describe an experiment without importing or executing
ready-made attacks, embedding methods, or metrics. Registry-dependent validation
and deterministic expansion belong to the catalog and planner layers.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1

StrictBool = Annotated[bool, Field(strict=True)]
StrictPositiveInt = Annotated[int, Field(strict=True, gt=0)]
StrictSeed = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
StrictFraction = Annotated[float, Field(strict=True, gt=0.0, le=1.0)]
StrictPrecision = Annotated[int, Field(strict=True, ge=0, le=15)]

NonEmptyText = Annotated[str, Field(strict=True, min_length=1, max_length=2048)]
ExperimentName = Annotated[str, Field(strict=True, min_length=1, max_length=128)]
ConfigId = Annotated[
    str,
    Field(
        strict=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    ),
]
SolutionName = Annotated[
    str,
    Field(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    ),
]
ParameterName = Annotated[
    str,
    Field(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    ),
]
ImportPath = Annotated[
    str,
    Field(
        strict=True,
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$",
    ),
]
GlobPattern = Annotated[str, Field(strict=True, min_length=1, max_length=512)]
FileExtension = Annotated[str, Field(strict=True, min_length=2, max_length=32)]

AutoOrPositiveInt = Union[Literal["auto"], StrictPositiveInt]

ArtifactReference = Literal[
    "original_image",
    "embedded_image",
    "attacked_image",
    "original_watermark",
    "extracted_watermark",
]
MetricCheckpoint = Literal["after_embedding", "after_attack", "after_extraction"]

_RUNTIME_PARAMETER_NAMES = frozenset(
    {
        "attacked_image",
        "distorted_image",
        "embedded_image",
        "extracted_bits",
        "extracted_watermark",
        "input_image",
        "num_bits",
        "original_bits",
        "original_image",
        "original_watermark",
        "output_image",
        "watermark",
        "watermark_bits",
    }
)

_AVAILABLE_ARTIFACTS = {
    "after_embedding": frozenset(
        {
            "original_image",
            "embedded_image",
            "original_watermark",
        }
    ),
    "after_attack": frozenset(
        {
            "original_image",
            "embedded_image",
            "attacked_image",
            "original_watermark",
        }
    ),
    "after_extraction": frozenset(
        {
            "original_image",
            "embedded_image",
            "attacked_image",
            "original_watermark",
            "extracted_watermark",
        }
    ),
}


class StrictConfigModel(BaseModel):
    """Base class shared by every experiment configuration model."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class _FrozenDict(dict):
    """Pickle-safe dictionary whose contents cannot change after creation."""

    def _blocked(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("configuration mappings are immutable")

    __setitem__ = _blocked
    __delitem__ = _blocked
    __ior__ = _blocked
    clear = _blocked
    pop = _blocked
    popitem = _blocked
    setdefault = _blocked
    update = _blocked

    def __copy__(self) -> _FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenDict:
        return self

    def __reduce__(self) -> tuple[type[_FrozenDict], tuple[dict[Any, Any]]]:
        return type(self), (dict(self),)


def _freeze_json_value(value: Any, location: str) -> Any:
    """Return a detached and immutable JSON-compatible representation."""

    if value is None or isinstance(value, (bool, str)):
        return value

    if isinstance(value, int) and not isinstance(value, bool):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must not contain NaN or infinity")
        return value

    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item, f"{location}[{index}]") for index, item in enumerate(value))

    if isinstance(value, Mapping):
        normalised = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} contains a non-string mapping key: {key!r}")
            normalised[key] = _freeze_json_value(item, f"{location}.{key}")
        return _FrozenDict(normalised)

    raise ValueError(
        f"{location} contains unsupported value of type {type(value).__name__}; only JSON-compatible values are allowed"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _ensure_unique(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    seen = set()
    duplicates = set()
    for item in values:
        if item in seen:
            duplicates.add(item)
        seen.add(item)

    if duplicates:
        joined = ", ".join(repr(item) for item in sorted(duplicates))
        raise ValueError(f"{field_name} contains duplicate values: {joined}")
    return values


def _ensure_non_empty_path(value: Any, field_name: str) -> Any:
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _resolve_path(path: Path, base_directory: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = base_directory / expanded
    return expanded.resolve(strict=False)


class ParameterSpace(StrictConfigModel):
    """Fixed values and one explicit parameter-variation strategy.

    ``fixed`` values are always applied. A Cartesian ``grid`` and explicit
    linked ``variants`` are deliberately mutually exclusive. This keeps the
    meaning of a configuration unambiguous when the planner expands it.
    """

    fixed: Mapping[ParameterName, Any] = Field(default_factory=dict)
    grid: Mapping[ParameterName, tuple[Any, ...]] = Field(default_factory=dict)
    variants: tuple[Mapping[ParameterName, Any], ...] = ()

    @field_validator("fixed")
    @classmethod
    def validate_fixed_values(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return _FrozenDict({name: _freeze_json_value(item, f"fixed.{name}") for name, item in value.items()})

    @field_validator("grid")
    @classmethod
    def validate_grid_values(
        cls,
        value: Mapping[str, tuple[Any, ...]],
    ) -> Mapping[str, tuple[Any, ...]]:
        normalised = {}
        for name, candidates in value.items():
            if not candidates:
                raise ValueError(f"grid.{name} must contain at least one candidate")

            items = tuple(_freeze_json_value(item, f"grid.{name}[{index}]") for index, item in enumerate(candidates))
            fingerprints = [_canonical_json(item) for item in items]
            if len(fingerprints) != len(set(fingerprints)):
                raise ValueError(f"grid.{name} contains duplicate candidates")
            normalised[name] = items
        return _FrozenDict(normalised)

    @field_validator("variants")
    @classmethod
    def validate_variant_values(
        cls,
        value: tuple[Mapping[str, Any], ...],
    ) -> tuple[Mapping[str, Any], ...]:
        normalised = []
        for variant_index, variant in enumerate(value):
            if not variant:
                raise ValueError(f"variants[{variant_index}] must not be empty")
            normalised.append(
                _FrozenDict(
                    {
                        name: _freeze_json_value(
                            item,
                            f"variants[{variant_index}].{name}",
                        )
                        for name, item in variant.items()
                    }
                )
            )
        return tuple(normalised)

    @model_validator(mode="after")
    def validate_parameter_space(self) -> ParameterSpace:
        if "variants" in self.model_fields_set and not self.variants:
            raise ValueError("variants must contain at least one parameter combination")

        if self.grid and self.variants:
            raise ValueError("grid and variants are mutually exclusive")

        variable_keys = set(self.grid)
        if self.variants:
            expected_keys = set(self.variants[0])
            for index, variant in enumerate(self.variants[1:], start=1):
                if set(variant) != expected_keys:
                    raise ValueError(
                        "every explicit variant must declare the same parameter keys; "
                        f"variants[0] has {sorted(expected_keys)!r}, "
                        f"variants[{index}] has {sorted(variant)!r}"
                    )
            variable_keys = expected_keys

            fingerprints = [_canonical_json(variant) for variant in self.variants]
            if len(fingerprints) != len(set(fingerprints)):
                raise ValueError("variants contains duplicate parameter combinations")

        overlap = set(self.fixed) & variable_keys
        if overlap:
            raise ValueError("parameters cannot be both fixed and variable: " + ", ".join(sorted(overlap)))

        all_keys = set(self.fixed) | variable_keys
        reserved = sorted(all_keys & _RUNTIME_PARAMETER_NAMES)
        if reserved:
            raise ValueError("runtime-managed parameters must not be configured by the user: " + ", ".join(reserved))

        return self

    @property
    def declared_variant_count(self) -> int:
        """Return the number of variants represented by this declaration."""

        if self.variants:
            return len(self.variants)
        if self.grid:
            count = 1
            for candidates in self.grid.values():
                count *= len(candidates)
            return count
        return 1


class OperationSpec(StrictConfigModel):
    """Parameter declaration for one pipeline operation."""

    params: ParameterSpace = Field(default_factory=ParameterSpace)


class ExperimentMetadata(StrictConfigModel):
    """Human-readable and reproducibility-related experiment settings."""

    name: ExperimentName
    description: Optional[NonEmptyText] = None
    seed: StrictSeed = 0
    repeats: StrictPositiveInt = 1
    max_cases: StrictPositiveInt = 1_000_000
    allow_large_plan: StrictBool = False
    tags: tuple[ConfigId, ...] = ()

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _ensure_unique(value, "experiment.tags")


class ImagePreprocessingSpec(StrictConfigModel):
    """Deterministic preprocessing requested from the dataset source."""

    color_mode: Literal["RGB"] = "RGB"
    resize: Optional[tuple[StrictPositiveInt, StrictPositiveInt]] = None
    resample: Literal["nearest", "bilinear", "bicubic", "lanczos"] = "lanczos"
    exif_transpose: StrictBool = True


class DirectoryDatasetSpec(StrictConfigModel):
    """Images discovered below one directory.

    ``resize`` uses the conventional ``(width, height)`` order. Path existence
    is deliberately not checked here, because validation must remain usable for
    dry-run tooling and configurations prepared on another machine.
    """

    type: Literal["directory"]
    path: Path
    recursive: StrictBool = True
    extensions: tuple[FileExtension, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    )
    include: tuple[GlobPattern, ...] = ()
    exclude: tuple[GlobPattern, ...] = ()
    limit: Optional[StrictPositiveInt] = None
    shuffle: StrictBool = False
    preprocessing: ImagePreprocessingSpec = Field(default_factory=ImagePreprocessingSpec)

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, value: Any) -> Any:
        return _ensure_non_empty_path(value, "dataset.path")

    @field_validator("extensions")
    @classmethod
    def validate_extensions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("dataset.extensions must not be empty")

        normalised = []
        for extension in value:
            extension = extension.strip().lower()
            if not re.fullmatch(r"\.[a-z0-9]+", extension):
                raise ValueError("every dataset extension must start with '.' and contain only letters or digits")
            normalised.append(extension)

        return _ensure_unique(tuple(normalised), "dataset.extensions")

    @field_validator("include", "exclude")
    @classmethod
    def validate_glob_patterns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _ensure_unique(value, "dataset glob patterns")

    @model_validator(mode="after")
    def validate_pattern_sets(self) -> DirectoryDatasetSpec:
        overlap = sorted(set(self.include) & set(self.exclude))
        if overlap:
            raise ValueError(
                "dataset include and exclude contain the same patterns: " + ", ".join(repr(item) for item in overlap)
            )
        return self


class RandomBitsWatermarkSpec(StrictConfigModel):
    """A random bit sequence derived deterministically from the run seed."""

    type: Literal["random_bits"]
    length: StrictPositiveInt
    scope: Literal["fixed_for_run", "per_image", "per_case"] = "fixed_for_run"


class FixedBitsWatermarkSpec(StrictConfigModel):
    """A literal binary sequence used by every applicable experiment case."""

    type: Literal["fixed_bits"]
    bits: Annotated[
        str,
        Field(
            strict=True,
            min_length=1,
            max_length=1_000_000,
            pattern=r"^[01]+$",
        ),
    ]


WatermarkSpec = Annotated[
    Union[RandomBitsWatermarkSpec, FixedBitsWatermarkSpec],
    Field(discriminator="type"),
]


class EmbeddingSpec(StrictConfigModel):
    """One named embedding solution and its two parameter spaces."""

    id: ConfigId
    name: SolutionName
    embedding: OperationSpec = Field(default_factory=OperationSpec)
    extraction: OperationSpec = Field(default_factory=OperationSpec)


class AttackStepSpec(StrictConfigModel):
    """One step in an ordered attack scenario."""

    name: SolutionName
    params: ParameterSpace = Field(default_factory=ParameterSpace)


class AttackScenarioSpec(StrictConfigModel):
    """An ordered attack chain; an empty chain is a clean baseline."""

    id: ConfigId
    description: Optional[NonEmptyText] = None
    steps: tuple[AttackStepSpec, ...] = ()


class MetricBindingSpec(StrictConfigModel):
    """A metric invocation bound to named pipeline artifacts."""

    id: ConfigId
    name: SolutionName
    checkpoint: MetricCheckpoint
    inputs: Mapping[ParameterName, ArtifactReference]
    params: ParameterSpace = Field(default_factory=ParameterSpace)

    @field_validator("inputs")
    @classmethod
    def validate_inputs(
        cls,
        value: Mapping[str, ArtifactReference],
    ) -> Mapping[str, ArtifactReference]:
        if not value:
            raise ValueError("metric inputs must not be empty")
        return _FrozenDict(value)

    @model_validator(mode="after")
    def validate_metric_binding(self) -> MetricBindingSpec:
        available = _AVAILABLE_ARTIFACTS[self.checkpoint]
        unavailable = sorted(set(self.inputs.values()) - available)
        if unavailable:
            raise ValueError(f"checkpoint {self.checkpoint!r} cannot access artifacts: " + ", ".join(unavailable))

        configured_parameters = set(self.params.fixed) | set(self.params.grid)
        if self.params.variants:
            configured_parameters |= set(self.params.variants[0])

        collisions = sorted(set(self.inputs) & configured_parameters)
        if collisions:
            raise ValueError(
                "metric input bindings and configured parameters must use distinct "
                "argument names: " + ", ".join(collisions)
            )
        return self


class ExecutionSpec(StrictConfigModel):
    """Execution settings for future serial and process executors."""

    backend: Literal["serial", "process"] = "serial"
    workers: AutoOrPositiveInt = "auto"
    max_in_flight: AutoOrPositiveInt = "auto"
    start_method: Literal["auto", "spawn", "fork", "forkserver"] = "auto"
    fail_fast: StrictBool = False
    show_progress: StrictBool = True

    @model_validator(mode="after")
    def validate_execution(self) -> ExecutionSpec:
        if self.backend == "serial":
            if isinstance(self.workers, int) and self.workers != 1:
                raise ValueError("serial execution requires workers=1 or workers='auto'")
            if isinstance(self.max_in_flight, int) and self.max_in_flight != 1:
                raise ValueError("serial execution requires max_in_flight=1 or max_in_flight='auto'")
            if self.start_method != "auto":
                raise ValueError("start_method is only applicable to process execution")

        if isinstance(self.workers, int) and isinstance(self.max_in_flight, int) and self.max_in_flight < self.workers:
            raise ValueError("max_in_flight must be greater than or equal to workers")

        return self


class OutputSpec(StrictConfigModel):
    """Persistent result and artifact settings."""

    directory: Path = Path("runs")
    database: Literal["sqlite"] = "sqlite"
    database_filename: Annotated[
        str,
        Field(strict=True, min_length=1, max_length=255),
    ] = "results.sqlite3"
    save_images: Literal["none", "failures", "sampled", "all"] = "failures"
    sample_fraction: Optional[StrictFraction] = None
    exports: tuple[Literal["csv", "jsonl"], ...] = ("csv",)
    overwrite: StrictBool = False

    @field_validator("sample_fraction", mode="before")
    @classmethod
    def validate_sample_fraction_type(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, float):
            raise ValueError("sample_fraction must be a floating-point number")
        return value

    @field_validator("directory", mode="before")
    @classmethod
    def validate_directory(cls, value: Any) -> Any:
        return _ensure_non_empty_path(value, "output.directory")

    @field_validator("database_filename")
    @classmethod
    def validate_database_filename(cls, value: str) -> str:
        if value in {".", ".."} or "/" in value or "\\" in value or Path(value).name != value:
            raise ValueError("database_filename must be a file name, not a path")
        if not value.lower().endswith((".db", ".sqlite", ".sqlite3")):
            raise ValueError("database_filename must end with .db, .sqlite, or .sqlite3")
        return value

    @field_validator("exports")
    @classmethod
    def validate_exports(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _ensure_unique(value, "output.exports")

    @model_validator(mode="after")
    def validate_image_sampling(self) -> OutputSpec:
        if self.save_images == "sampled" and self.sample_fraction is None:
            raise ValueError("sample_fraction is required when save_images='sampled'")
        if self.save_images != "sampled" and self.sample_fraction is not None:
            raise ValueError("sample_fraction is only allowed when save_images='sampled'")
        return self


class ReportSpec(StrictConfigModel):
    """Rules for reports generated from persisted experiment results."""

    enabled: StrictBool = True
    formats: tuple[Literal["csv", "html"], ...] = ("csv",)
    group_by: tuple[
        Literal[
            "sample",
            "embedding",
            "embedding_params",
            "attack_scenario",
            "attack_params",
            "metric",
        ],
        ...,
    ] = ("embedding", "attack_scenario", "metric")
    include_failures: StrictBool = True
    float_precision: StrictPrecision = 6

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _ensure_unique(value, "report.formats")

    @field_validator("group_by")
    @classmethod
    def validate_group_by(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _ensure_unique(value, "report.group_by")

    @model_validator(mode="after")
    def validate_report(self) -> ReportSpec:
        if self.enabled and not self.formats:
            raise ValueError("report.formats must not be empty when reporting is enabled")
        return self


class ExperimentConfig(StrictConfigModel):
    """Complete schema-v1 declaration of a DWARF experiment."""

    schema_version: Literal[1]
    experiment: ExperimentMetadata
    plugins: tuple[ImportPath, ...] = ()
    dataset: DirectoryDatasetSpec
    watermark: WatermarkSpec
    embeddings: Annotated[tuple[EmbeddingSpec, ...], Field(min_length=1)]
    attack_scenarios: Annotated[
        tuple[AttackScenarioSpec, ...],
        Field(min_length=1),
    ] = Field(default_factory=lambda: (AttackScenarioSpec(id="clean"),))
    metrics: tuple[MetricBindingSpec, ...] = ()
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)
    report: ReportSpec = Field(default_factory=ReportSpec)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: Any) -> Any:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("schema_version must be the integer 1")
        if value != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {value!r}; expected {SCHEMA_VERSION}")
        return value

    @field_validator("plugins")
    @classmethod
    def validate_plugins(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _ensure_unique(value, "plugins")

    @model_validator(mode="after")
    def validate_unique_ids(self) -> ExperimentConfig:
        _ensure_unique(
            tuple(item.id for item in self.embeddings),
            "embeddings ids",
        )
        _ensure_unique(
            tuple(item.id for item in self.attack_scenarios),
            "attack_scenarios ids",
        )
        _ensure_unique(
            tuple(item.id for item in self.metrics),
            "metrics ids",
        )
        return self

    def resolve_paths(self, base_directory: Union[str, Path]) -> ExperimentConfig:
        """Return a copy with configuration-owned paths made absolute."""

        base_directory = Path(base_directory).expanduser().resolve(strict=False)
        dataset = self.dataset.model_copy(update={"path": _resolve_path(self.dataset.path, base_directory)})
        output = self.output.model_copy(update={"directory": _resolve_path(self.output.directory, base_directory)})
        return self.model_copy(update={"dataset": dataset, "output": output})


__all__ = [
    "SCHEMA_VERSION",
    "ArtifactReference",
    "AttackScenarioSpec",
    "AttackStepSpec",
    "DirectoryDatasetSpec",
    "EmbeddingSpec",
    "ExecutionSpec",
    "ExperimentConfig",
    "ExperimentMetadata",
    "FixedBitsWatermarkSpec",
    "ImagePreprocessingSpec",
    "MetricBindingSpec",
    "OperationSpec",
    "OutputSpec",
    "ParameterSpace",
    "RandomBitsWatermarkSpec",
    "ReportSpec",
    "WatermarkSpec",
]
