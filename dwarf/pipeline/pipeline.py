# ruff: noqa: UP045
"""High-level facade for validating, planning and serially executing experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from dwarf.pipeline.catalog import SolutionCatalog
from dwarf.pipeline.config import ExperimentConfig
from dwarf.pipeline.config_loader import load_experiment_config
from dwarf.pipeline.datasets import DirectoryDatasetSource
from dwarf.pipeline.executors import SerialExecutor, UnsupportedExecutionBackendError
from dwarf.pipeline.planner import ExperimentPlan, ExperimentPlanner
from dwarf.pipeline.results import ExecutionResult
from dwarf.pipeline.runtime import ErasurePolicy, RuntimeAdapterRegistry


class Pipeline:
    """Lazy facade joining all currently implemented experiment pipeline stages."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        catalog: Optional[SolutionCatalog] = None,
        adapters: Optional[RuntimeAdapterRegistry] = None,
    ) -> None:
        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig")
        self.config = config
        self._catalog = catalog
        self.adapters = RuntimeAdapterRegistry() if adapters is None else adapters
        self._resolved = None
        self._dataset_source = None
        self._manifest = None
        self._plan = None

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],  # noqa: UP007
        *,
        resolve_paths: bool = True,
        catalog: Optional[SolutionCatalog] = None,
        adapters: Optional[RuntimeAdapterRegistry] = None,
    ) -> Pipeline:
        """Load a YAML configuration and create a lazy pipeline facade."""

        return cls(
            load_experiment_config(path, resolve_paths=resolve_paths),
            catalog=catalog,
            adapters=adapters,
        )

    @property
    def catalog(self) -> SolutionCatalog:
        if self._catalog is None:
            self._catalog = SolutionCatalog.discover(plugins=self.config.plugins)
        return self._catalog

    @property
    def resolved(self):
        if self._resolved is None:
            self._resolved = self.catalog.resolve(self.config)
        return self._resolved

    @property
    def dataset_source(self) -> DirectoryDatasetSource:
        if self._dataset_source is None:
            self._dataset_source = DirectoryDatasetSource(self.config.dataset)
        return self._dataset_source

    @property
    def manifest(self):
        if self._manifest is None:
            self._manifest = self.dataset_source.build_manifest(seed=self.config.experiment.seed)
        return self._manifest

    def validate(self):
        """Run structural and semantic validation and return the resolved experiment."""

        return self.resolved

    def plan(self) -> ExperimentPlan:
        """Build and cache a deterministic experiment plan."""

        if self._plan is None:
            self._plan = ExperimentPlanner(self.resolved, self.manifest).build()
        return self._plan

    def run(
        self,
        *,
        erasure_policy: ErasurePolicy = ErasurePolicy.COUNT_AS_ERROR,
        verify_integrity: bool = True,
        fail_fast: Optional[bool] = None,
    ) -> ExecutionResult:
        """Execute the prepared plan with the configured backend."""

        backend = self.config.execution.backend
        if backend != "serial":
            raise UnsupportedExecutionBackendError(
                f"execution backend {backend!r} is not implemented; use backend='serial'"
            )
        executor = SerialExecutor(
            self.plan(),
            self.resolved,
            self.dataset_source,
            adapters=self.adapters,
            erasure_policy=erasure_policy,
            verify_integrity=verify_integrity,
            fail_fast=fail_fast,
        )
        return executor.execute()

    def reset_runtime_state(self) -> None:
        """Discard the cached manifest and plan while retaining validated metadata."""

        self._manifest = None
        self._plan = None


__all__ = ["Pipeline"]
