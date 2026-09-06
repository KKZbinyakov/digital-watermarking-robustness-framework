# ruff: noqa: UP045
"""Immutable records describing the files selected from an image dataset."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from dwarf.pipeline.exceptions import DatasetManifestError

MANIFEST_SCHEMA_VERSION = 1
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _strict_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetManifestError(f"{field_name} must be a non-negative integer")
    return value


def _strict_positive_int_or_none(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatasetManifestError(f"{field_name} must be a positive integer or null")
    return value


def _normalise_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise DatasetManifestError(f"{field_name} must be a hexadecimal SHA-256 string")
    normalised = value.lower()
    if not _HEX_SHA256.fullmatch(normalised):
        raise DatasetManifestError(f"{field_name} must contain exactly 64 hexadecimal characters")
    return normalised


def _normalise_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetManifestError("relative_path must be a non-empty POSIX path")
    if "\0" in value:
        raise DatasetManifestError("relative_path must not contain null characters")

    normalised = value.replace("\\", "/")
    path = PurePosixPath(normalised)
    if path.is_absolute() or not path.parts:
        raise DatasetManifestError("relative_path must be relative to the dataset root")
    if re.fullmatch(r"[A-Za-z]:", path.parts[0]):
        raise DatasetManifestError("relative_path must not contain a Windows drive prefix")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise DatasetManifestError("relative_path must not contain empty, '.' or '..' components")
    return path.as_posix()


def _normalise_string_tuple(values: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise DatasetManifestError(f"{field_name} must be a sequence of strings")
    try:
        result = tuple(values)
    except TypeError as error:
        raise DatasetManifestError(f"{field_name} must be a sequence of strings") from error
    if any(not isinstance(item, str) or not item for item in result):
        raise DatasetManifestError(f"{field_name} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise DatasetManifestError(f"{field_name} must not contain duplicate values")
    return result


def _normalise_absolute_path(value: Any, field_name: str) -> Path:
    try:
        path = Path(value).expanduser()
    except (TypeError, OSError, RuntimeError) as error:
        raise DatasetManifestError(f"{field_name} must be a valid absolute path") from error
    if not path.is_absolute():
        raise DatasetManifestError(f"{field_name} must be absolute")
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise DatasetManifestError(f"{field_name} could not be normalised: {error}") from error


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def make_sample_id(relative_path: str, checksum_sha256: str) -> str:
    """Return a stable identifier derived from path and file content."""

    relative_path = _normalise_relative_path(relative_path)
    checksum_sha256 = _normalise_sha256(checksum_sha256, "checksum_sha256")
    payload = relative_path.encode("utf-8") + b"\0" + checksum_sha256.encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SampleReference:
    """A small, pickle-safe reference to one immutable manifest entry."""

    sample_id: str
    relative_path: str
    path: Path
    size_bytes: int
    modified_time_ns: int
    checksum_sha256: str

    def __post_init__(self) -> None:
        sample_id = _normalise_sha256(self.sample_id, "sample_id")
        relative_path = _normalise_relative_path(self.relative_path)
        checksum = _normalise_sha256(self.checksum_sha256, "checksum_sha256")
        path = _normalise_absolute_path(self.path, "sample path")

        size_bytes = _strict_non_negative_int(self.size_bytes, "size_bytes")
        modified_time_ns = _strict_non_negative_int(self.modified_time_ns, "modified_time_ns")
        expected_sample_id = make_sample_id(relative_path, checksum)
        if sample_id != expected_sample_id:
            raise DatasetManifestError(
                "sample_id does not match the declared relative path and checksum",
            )

        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "relative_path", relative_path)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "size_bytes", size_bytes)
        object.__setattr__(self, "modified_time_ns", modified_time_ns)
        object.__setattr__(self, "checksum_sha256", checksum)

    @property
    def name(self) -> str:
        return PurePosixPath(self.relative_path).name

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.relative_path).suffix.lower()

    def to_dict(self, *, include_absolute_path: bool = True) -> dict[str, Any]:
        """Return a JSON-compatible record."""

        result = {
            "sample_id": self.sample_id,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "modified_time_ns": self.modified_time_ns,
            "checksum_sha256": self.checksum_sha256,
        }
        if include_absolute_path:
            result["path"] = str(self.path)
        return result


@dataclass(frozen=True)
class DatasetManifest:
    """Deterministic ordered snapshot of selected dataset files.

    The fingerprint intentionally excludes the absolute root, file mtimes and
    selection procedure. Moving the same ordered byte-identical files to
    another machine therefore preserves the fingerprint; changes to selected
    paths, bytes or order change it.
    """

    root: Path
    samples: tuple[SampleReference, ...]
    recursive: bool
    extensions: tuple[str, ...]
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    shuffled: bool
    seed: int
    limit: Optional[int]
    schema_version: int = field(default=MANIFEST_SCHEMA_VERSION, init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        root = _normalise_absolute_path(self.root, "manifest root")
        if not isinstance(self.recursive, bool):
            raise DatasetManifestError("recursive must be a boolean")
        if not isinstance(self.shuffled, bool):
            raise DatasetManifestError("shuffled must be a boolean")

        raw_extensions = _normalise_string_tuple(self.extensions, "extensions")
        extensions = _normalise_string_tuple(
            tuple(sorted(item.lower() for item in raw_extensions)),
            "extensions",
        )
        include = tuple(sorted(_normalise_string_tuple(self.include, "include")))
        exclude = tuple(sorted(_normalise_string_tuple(self.exclude, "exclude")))
        seed = _strict_non_negative_int(self.seed, "seed")
        limit = _strict_positive_int_or_none(self.limit, "limit")
        try:
            samples = tuple(self.samples)
        except TypeError as error:
            raise DatasetManifestError("samples must be an iterable of SampleReference values") from error
        if not samples:
            raise DatasetManifestError("a dataset manifest must contain at least one sample")
        if any(not isinstance(sample, SampleReference) for sample in samples):
            raise DatasetManifestError("samples must contain only SampleReference values")

        sample_ids = [sample.sample_id for sample in samples]
        relative_paths = [sample.relative_path for sample in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise DatasetManifestError("manifest contains duplicate sample IDs")
        if len(relative_paths) != len(set(relative_paths)):
            raise DatasetManifestError("manifest contains duplicate relative paths")

        for sample in samples:
            if not _path_is_within(sample.path, root):
                raise DatasetManifestError(
                    f"sample path {sample.path} is outside manifest root {root}",
                )
            expected_path = root.joinpath(*PurePosixPath(sample.relative_path).parts)
            if sample.path != expected_path:
                raise DatasetManifestError(
                    f"sample path {sample.path} does not match relative path {sample.relative_path!r}"
                )

        object.__setattr__(self, "root", root)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "extensions", extensions)
        object.__setattr__(self, "include", include)
        object.__setattr__(self, "exclude", exclude)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "fingerprint", self._calculate_fingerprint())

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "relative_path": sample.relative_path,
                    "size_bytes": sample.size_bytes,
                    "checksum_sha256": sample.checksum_sha256,
                }
                for sample in self.samples
            ],
        }

    def _calculate_fingerprint(self) -> str:
        encoded = json.dumps(
            self._fingerprint_payload(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def by_id(self, sample_id: str) -> SampleReference:
        """Return one sample or raise ``KeyError`` for an unknown identifier."""

        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        raise KeyError(sample_id)

    def to_dict(self, *, include_absolute_paths: bool = True) -> dict[str, Any]:
        """Return a JSON-compatible manifest representation."""

        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "root": str(self.root),
            "selection": {
                "recursive": self.recursive,
                "extensions": list(self.extensions),
                "include": list(self.include),
                "exclude": list(self.exclude),
                "shuffled": self.shuffled,
                "seed": self.seed,
                "limit": self.limit,
            },
            "samples": [sample.to_dict(include_absolute_path=include_absolute_paths) for sample in self.samples],
        }

    def to_json(self, *, indent: Optional[int] = 2, include_absolute_paths: bool = True) -> str:
        """Serialise the manifest deterministically as UTF-8 JSON text."""

        return json.dumps(
            self.to_dict(include_absolute_paths=include_absolute_paths),
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[SampleReference]:
        return iter(self.samples)


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "DatasetManifest",
    "SampleReference",
    "make_sample_id",
]
