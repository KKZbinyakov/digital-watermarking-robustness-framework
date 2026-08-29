# ruff: noqa: UP045
"""Exceptions raised while reading experiment configuration files."""

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


__all__ = ["ConfigLoadError"]
