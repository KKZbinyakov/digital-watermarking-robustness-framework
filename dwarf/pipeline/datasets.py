"""Deterministic directory discovery and image loading for pipeline datasets."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from dwarf.pipeline.artifacts import ImageArtifact
from dwarf.pipeline.config import DirectoryDatasetSpec
from dwarf.pipeline.exceptions import (
    DatasetDiscoveryError,
    DatasetIntegrityError,
    DatasetLoadError,
    DatasetManifestError,
)
from dwarf.pipeline.manifest import DatasetManifest, SampleReference, make_sample_id

_HASH_CHUNK_SIZE = 1024 * 1024
_MAX_SEED = 2**63 - 1
_RESAMPLING_FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed > _MAX_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_SEED}")
    return seed


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _normalise_glob(pattern: str) -> str:
    normalised = pattern.replace("\\", "/")
    while normalised.startswith("./"):
        normalised = normalised[2:]
    return normalised


def _matches_glob(relative_path: str, pattern: str) -> bool:
    path = PurePosixPath(relative_path)
    pattern = _normalise_glob(pattern)
    if path.match(pattern):
        return True
    return pattern.startswith("**/") and path.match(pattern[3:])


def _matches_any(relative_path: str, patterns: tuple[str, ...]) -> bool:
    return any(_matches_glob(relative_path, pattern) for pattern in patterns)


def _stable_shuffle_key(relative_path: str, seed: int) -> tuple[bytes, str, str]:
    payload = str(seed).encode("ascii") + b"\0" + relative_path.encode("utf-8")
    return hashlib.sha256(payload).digest(), relative_path.casefold(), relative_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_file_with_sha256(path: Path) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    chunks = []
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
            chunks.append(chunk)
    return b"".join(chunks), digest.hexdigest()


def _contains_symlink(root: Path, relative_path: str) -> bool:
    current = root
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


class DirectoryDatasetSource:
    """Discover and load images described by :class:`DirectoryDatasetSpec`.

    The source owns no queues and performs no multiprocessing. A manifest
    contains only small file references, so future executors can send those
    references to workers and decode each image inside the worker process.
    """

    __slots__ = ("_configured_root", "spec")

    def __init__(self, spec: DirectoryDatasetSpec) -> None:
        if not isinstance(spec, DirectoryDatasetSpec):
            raise TypeError(
                f"spec must be a DirectoryDatasetSpec, got {type(spec).__name__}",
            )
        self.spec = spec
        self._configured_root = spec.path.expanduser().resolve(strict=False)

    @property
    def root(self) -> Path:
        """Return the bound absolute root without requiring it to exist."""

        return self._configured_root

    def _require_root(self) -> Path:
        configured = self._configured_root
        try:
            root = configured.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise DatasetDiscoveryError(
                f"dataset path {configured} does not exist or cannot be resolved: {error}",
            ) from error
        if not root.is_dir():
            raise DatasetDiscoveryError(f"dataset path {root} is not a directory")
        return root

    def _candidate_paths(self, root: Path) -> list[tuple[str, Path]]:
        candidates = []

        if self.spec.recursive:

            def raise_walk_error(error: OSError) -> None:
                raise DatasetDiscoveryError(
                    f"failed to scan dataset directory {error.filename!r}: {error}",
                ) from error

            for directory, directory_names, file_names in os.walk(
                root,
                topdown=True,
                onerror=raise_walk_error,
                followlinks=False,
            ):
                directory_path = Path(directory)
                directory_names[:] = [
                    name
                    for name in sorted(directory_names, key=lambda item: (item.casefold(), item))
                    if not (directory_path / name).is_symlink()
                ]
                for file_name in sorted(file_names, key=lambda item: (item.casefold(), item)):
                    self._append_candidate(root, directory_path / file_name, candidates)
        else:
            try:
                entries = sorted(
                    root.iterdir(),
                    key=lambda item: (item.name.casefold(), item.name),
                )
            except OSError as error:
                raise DatasetDiscoveryError(
                    f"failed to scan dataset directory {root}: {error}",
                ) from error
            for candidate in entries:
                self._append_candidate(root, candidate, candidates)

        candidates.sort(key=lambda item: (item[0].casefold(), item[0]))
        return candidates

    def _append_candidate(
        self,
        root: Path,
        candidate: Path,
        candidates: list[tuple[str, Path]],
    ) -> None:
        try:
            if candidate.is_symlink() or not candidate.is_file():
                return
        except OSError as error:
            raise DatasetDiscoveryError(
                f"failed to inspect dataset entry {candidate}: {error}",
            ) from error

        if candidate.suffix.lower() not in self.spec.extensions:
            return

        try:
            relative_path = candidate.relative_to(root).as_posix()
        except ValueError as error:
            raise DatasetDiscoveryError(
                f"dataset entry {candidate} is outside root {root}",
            ) from error

        if self.spec.include and not _matches_any(relative_path, self.spec.include):
            return
        if self.spec.exclude and _matches_any(relative_path, self.spec.exclude):
            return
        candidates.append((relative_path, candidate))

    def _build_reference(
        self,
        root: Path,
        relative_path: str,
        candidate: Path,
    ) -> SampleReference:
        if _contains_symlink(root, relative_path):
            raise DatasetIntegrityError(
                f"dataset entry {relative_path!r} became a symbolic link during discovery",
            )

        try:
            resolved = candidate.resolve(strict=True)
            if not _path_is_within(resolved, root):
                raise DatasetIntegrityError(
                    f"dataset entry {relative_path!r} resolves outside root {root}",
                )
            before = resolved.stat()
            checksum = _sha256_file(resolved)
            after = resolved.stat()
        except DatasetIntegrityError:
            raise
        except OSError as error:
            raise DatasetDiscoveryError(
                f"failed to read dataset entry {relative_path!r}: {error}",
            ) from error

        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise DatasetIntegrityError(
                f"dataset entry {relative_path!r} changed while its manifest record was created"
            )

        return SampleReference(
            sample_id=make_sample_id(relative_path, checksum),
            relative_path=relative_path,
            path=resolved,
            size_bytes=after.st_size,
            modified_time_ns=after.st_mtime_ns,
            checksum_sha256=checksum,
        )

    def build_manifest(self, *, seed: int = 0) -> DatasetManifest:
        """Discover selected files and return a deterministic immutable manifest."""

        seed = _validate_seed(seed)
        root = self._require_root()
        candidates = self._candidate_paths(root)

        if self.spec.shuffle:
            candidates.sort(key=lambda item: _stable_shuffle_key(item[0], seed))
        if self.spec.limit is not None:
            candidates = candidates[: self.spec.limit]
        if not candidates:
            raise DatasetDiscoveryError(
                f"no image files matched the dataset configuration below {root}",
            )

        samples = tuple(
            self._build_reference(root, relative_path, candidate) for relative_path, candidate in candidates
        )
        return DatasetManifest(
            root=root,
            samples=samples,
            recursive=self.spec.recursive,
            extensions=self.spec.extensions,
            include=self.spec.include,
            exclude=self.spec.exclude,
            shuffled=self.spec.shuffle,
            seed=seed,
            limit=self.spec.limit,
        )

    def _resolve_sample_path(self, root: Path, sample: SampleReference) -> Path:
        relative_path = sample.relative_path
        if _contains_symlink(root, relative_path):
            raise DatasetIntegrityError(
                f"dataset entry {relative_path!r} contains a symbolic-link component",
            )

        lexical_path = root.joinpath(*PurePosixPath(relative_path).parts)
        try:
            resolved = lexical_path.resolve(strict=True)
        except OSError as error:
            raise DatasetIntegrityError(
                f"dataset entry {relative_path!r} no longer exists or cannot be resolved: {error}"
            ) from error

        if not _path_is_within(resolved, root):
            raise DatasetIntegrityError(
                f"dataset entry {relative_path!r} resolves outside root {root}",
            )
        if resolved != sample.path:
            raise DatasetIntegrityError(
                f"sample {sample.sample_id} does not belong to dataset root {root}",
            )
        if not resolved.is_file():
            raise DatasetIntegrityError(
                f"dataset entry {relative_path!r} is no longer a regular file",
            )
        return resolved

    def _verified_bytes(self, path: Path, sample: SampleReference) -> bytes:
        try:
            before = path.stat()
            if before.st_size != sample.size_bytes:
                raise DatasetIntegrityError(
                    f"dataset entry {sample.relative_path!r} has size {before.st_size}, expected {sample.size_bytes}"
                )
            data, checksum = _read_file_with_sha256(path)
            after = path.stat()
        except DatasetIntegrityError:
            raise
        except OSError as error:
            raise DatasetIntegrityError(
                f"failed to verify dataset entry {sample.relative_path!r}: {error}",
            ) from error

        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise DatasetIntegrityError(
                f"dataset entry {sample.relative_path!r} changed while it was verified",
            )
        if checksum != sample.checksum_sha256:
            raise DatasetIntegrityError(
                f"dataset entry {sample.relative_path!r} does not match its manifest checksum",
            )
        return data

    def verify(self, sample: SampleReference) -> None:
        """Verify that a sample still contains the bytes recorded in its manifest."""

        if not isinstance(sample, SampleReference):
            raise TypeError(
                f"sample must be a SampleReference, got {type(sample).__name__}",
            )
        root = self._require_root()
        path = self._resolve_sample_path(root, sample)
        self._verified_bytes(path, sample)

    def load(
        self,
        sample: SampleReference,
        *,
        verify_integrity: bool = True,
    ) -> ImageArtifact:
        """Decode, preprocess and return one canonical RGB image artifact."""

        if not isinstance(sample, SampleReference):
            raise TypeError(
                f"sample must be a SampleReference, got {type(sample).__name__}",
            )
        if not isinstance(verify_integrity, bool):
            raise TypeError("verify_integrity must be a boolean")

        root = self._require_root()
        path = self._resolve_sample_path(root, sample)
        source = BytesIO(self._verified_bytes(path, sample)) if verify_integrity else path

        preprocessing = self.spec.preprocessing
        try:
            with Image.open(source) as opened:
                opened.load()
                image = ImageOps.exif_transpose(opened) if preprocessing.exif_transpose else opened.copy()
                image = image.convert("RGB")
                if preprocessing.resize is not None:
                    image = image.resize(
                        preprocessing.resize,
                        resample=_RESAMPLING_FILTERS[preprocessing.resample],
                    )
                array = np.asarray(image, dtype=np.uint8)
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as error:
            raise DatasetLoadError(
                f"failed to decode dataset entry {sample.relative_path!r}: {error}",
            ) from error
        finally:
            if isinstance(source, BytesIO):
                source.close()

        return ImageArtifact(array)

    def iter_loaded(
        self,
        manifest: DatasetManifest,
        *,
        verify_integrity: bool = True,
    ) -> Iterator[tuple[SampleReference, ImageArtifact]]:
        """Yield manifest references paired with lazily decoded image artifacts."""

        if not isinstance(manifest, DatasetManifest):
            raise TypeError(
                f"manifest must be a DatasetManifest, got {type(manifest).__name__}",
            )
        root = self._require_root()
        if manifest.root != root:
            raise DatasetManifestError(
                f"manifest root {manifest.root} does not match dataset root {root}",
            )
        for sample in manifest:
            yield sample, self.load(sample, verify_integrity=verify_integrity)


__all__ = ["DirectoryDatasetSource"]
