# ruff: noqa: UP007
"""Safe YAML loading for strict DWARF experiment configurations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Union

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from dwarf.pipeline.config import ExperimentConfig
from dwarf.pipeline.exceptions import ConfigLoadError


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader with unique keys and YAML 1.2-style booleans."""


_StrictSafeLoader.yaml_implicit_resolvers = {
    character: resolvers.copy() for character, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
for first_character, resolvers in _StrictSafeLoader.yaml_implicit_resolvers.items():
    _StrictSafeLoader.yaml_implicit_resolvers[first_character] = [
        (tag, pattern) for tag, pattern in resolvers if tag != "tag:yaml.org,2002:bool"
    ]
_StrictSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


def _construct_unique_mapping(
    loader: _StrictSafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    if not isinstance(node, MappingNode):
        raise ConstructorError(
            None,
            None,
            f"expected a mapping node, got {node.id}",
            node.start_mark,
        )

    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            hash(key)
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found unhashable key {key!r}",
                key_node.start_mark,
            ) from error

        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_experiment_config(
    path: Union[str, Path],
    *,
    resolve_paths: bool = True,
) -> ExperimentConfig:
    """Load and validate one YAML experiment configuration.

    File and YAML syntax errors are wrapped in :class:`ConfigLoadError`.
    Pydantic's structured ``ValidationError`` is intentionally left unchanged
    for schema errors so callers can inspect exact field locations.
    """

    config_path = Path(path).expanduser()
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigLoadError(
            "configuration file must use the .yaml or .yml extension",
            path=config_path,
        )

    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigLoadError(
            "configuration file does not exist",
            path=config_path,
        ) from error
    except OSError as error:
        raise ConfigLoadError(
            f"could not read configuration file: {error}",
            path=config_path,
        ) from error

    try:
        raw = yaml.load(text, Loader=_StrictSafeLoader)
    except yaml.YAMLError as error:
        raise ConfigLoadError(f"invalid YAML: {error}", path=config_path) from error

    if raw is None:
        raise ConfigLoadError("configuration file is empty", path=config_path)
    if not isinstance(raw, dict):
        raise ConfigLoadError(
            "the YAML document root must be a mapping",
            path=config_path,
        )

    config = ExperimentConfig.model_validate(raw)
    if resolve_paths:
        config = config.resolve_paths(config_path.parent)
    return config


__all__ = ["load_experiment_config"]
