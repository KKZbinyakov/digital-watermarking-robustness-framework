# ruff: noqa: UP045
"""Exceptions raised by experiment configuration, discovery and datasets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class ConfigLoadError(ValueError):
    """A YAML configuration could not be read before schema validation."""

    def __init__(
        self,
        message: str,
        *,
        path: Optional[Path] = None,
    ) -> None:
        self.path = path
        prefix = f"{path}: " if path is not None else ""
        super().__init__(prefix + message)


class ArtifactValidationError(ValueError):
    """A value does not satisfy a canonical pipeline artifact contract."""


class DatasetSourceError(RuntimeError):
    """Base class for dataset discovery, manifest and loading failures."""


class DatasetManifestError(DatasetSourceError, ValueError):
    """A manifest or sample reference is internally inconsistent."""


class DatasetDiscoveryError(DatasetSourceError):
    """A directory dataset could not be discovered deterministically."""


class DatasetLoadError(DatasetSourceError):
    """A selected dataset image could not be decoded or normalised."""


class DatasetIntegrityError(DatasetSourceError):
    """A selected file changed or no longer matches its manifest entry."""


class SolutionCatalogError(RuntimeError):
    """Base class for failures while constructing a solution catalog."""


class SolutionDiscoveryError(SolutionCatalogError):
    """Built-in or plugin solutions could not be imported or described."""


class SolutionConflictError(SolutionCatalogError):
    """A plugin attempted to replace a previously registered solution."""


@dataclass(frozen=True)
class SemanticValidationIssue:
    """One independently actionable semantic problem in a configuration."""

    path: str
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path} [{self.code}]: {self.message}"


class SemanticValidationError(ValueError):
    """One or more registry-dependent validation checks failed."""

    def __init__(self, issues: Iterable[SemanticValidationIssue]) -> None:
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("SemanticValidationError requires at least one issue")
        suffix = "issue" if len(self.issues) == 1 else "issues"
        details = "\n".join(f"  - {issue}" for issue in self.issues)
        super().__init__(
            f"experiment configuration failed semantic validation with {len(self.issues)} {suffix}:\n{details}"
        )


__all__ = [
    "ArtifactValidationError",
    "ConfigLoadError",
    "DatasetDiscoveryError",
    "DatasetIntegrityError",
    "DatasetLoadError",
    "DatasetManifestError",
    "DatasetSourceError",
    "SemanticValidationError",
    "SemanticValidationIssue",
    "SolutionCatalogError",
    "SolutionConflictError",
    "SolutionDiscoveryError",
]
