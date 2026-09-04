# ruff: noqa: UP045
"""Immutable objects produced after solution discovery and semantic validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

from dwarf.pipeline.config import ExperimentConfig, MetricCheckpoint, OperationSpec, ParameterSpace
from dwarf.pipeline.solution_spec import SolutionSpec


@dataclass(frozen=True)
class ResolvedEmbedding:
    """One validated embedding declaration bound to a registered class."""

    id: str
    name: str
    implementation: type
    spec: SolutionSpec
    embedding: OperationSpec
    extraction: OperationSpec


@dataclass(frozen=True)
class ResolvedAttackStep:
    """One validated attack step bound to a registered class."""

    index: int
    name: str
    implementation: type
    spec: SolutionSpec
    params: ParameterSpace


@dataclass(frozen=True)
class ResolvedAttackScenario:
    """An ordered sequence of validated attack steps."""

    id: str
    description: Optional[str]
    steps: tuple[ResolvedAttackStep, ...]


@dataclass(frozen=True)
class ResolvedMetric:
    """One validated metric binding and its registered implementation."""

    id: str
    name: str
    implementation: type
    spec: SolutionSpec
    checkpoint: MetricCheckpoint
    inputs: Mapping[str, str]
    params: ParameterSpace


@dataclass(frozen=True)
class ResolvedExperiment:
    """A structurally and semantically validated experiment declaration."""

    config: ExperimentConfig
    embeddings: tuple[ResolvedEmbedding, ...]
    attack_scenarios: tuple[ResolvedAttackScenario, ...]
    metrics: tuple[ResolvedMetric, ...]
    plugins: tuple[str, ...]
    catalog_fingerprint: str


__all__ = [
    "ResolvedAttackScenario",
    "ResolvedAttackStep",
    "ResolvedEmbedding",
    "ResolvedExperiment",
    "ResolvedMetric",
]
