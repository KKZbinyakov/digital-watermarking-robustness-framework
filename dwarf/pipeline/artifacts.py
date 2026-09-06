# ruff: noqa: UP045
"""Canonical in-memory artifacts exchanged by future pipeline stages.

The wrappers in this module establish the data boundary used by dataset
sources, solution adapters, executors and metric bindings. Construction is
strict about semantic shape and values, while normalising ownership and memory
layout so that downstream code receives predictable NumPy arrays.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Optional

import numpy as np
from PIL import Image

from dwarf.pipeline.exceptions import ArtifactValidationError
from dwarf.pipeline.solution_spec import DataContract


def _require_ndarray(value: Any, artifact_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ArtifactValidationError(
            f"{artifact_name} must be constructed from a numpy.ndarray, got {type(value).__name__}"
        )
    return value


def _owned_c_array(value: np.ndarray, dtype: np.dtype[Any], *, writable: bool) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True, subok=False)
    result.setflags(write=writable)
    return result


def _normalise_integral_vector(
    value: Any,
    *,
    artifact_name: str,
    dtype: np.dtype[Any],
    allowed_values: frozenset[int],
) -> np.ndarray:
    if isinstance(value, str):
        raise ArtifactValidationError(
            f"{artifact_name} must be constructed from an integer sequence, not a string",
        )

    try:
        source = value if isinstance(value, np.ndarray) else tuple(value)
        array = np.asarray(source)
    except Exception as error:
        raise ArtifactValidationError(
            f"{artifact_name} could not be converted to a NumPy array: {error}",
        ) from error

    if array.ndim != 1:
        raise ArtifactValidationError(
            f"{artifact_name} must be one-dimensional, got shape {array.shape}",
        )
    if array.size == 0:
        raise ArtifactValidationError(f"{artifact_name} must contain at least one symbol")
    if array.dtype.kind not in {"b", "i", "u"}:
        raise ArtifactValidationError(
            f"{artifact_name} must contain integer symbols, got dtype {array.dtype}",
        )

    values = {int(item) for item in np.unique(array)}
    invalid = sorted(values - allowed_values)
    if invalid:
        allowed = ", ".join(str(item) for item in sorted(allowed_values))
        rejected = ", ".join(str(item) for item in invalid)
        raise ArtifactValidationError(
            f"{artifact_name} contains invalid symbols {rejected}; allowed values are {allowed}"
        )

    return _owned_c_array(array, dtype, writable=False)


class ImageArtifact:
    """Owned writable RGB image satisfying the ``rgb_uint8`` contract.

    The constructor deliberately copies the input. An artifact therefore owns
    its storage, is C-contiguous and writable, and does not alias the array from
    which it was created. The array remains mutable because existing DWARF
    solutions operate on NumPy arrays; callers must use :meth:`copy` when two
    independent pipeline branches need separate image state.
    """

    contract = DataContract.RGB_UINT8

    __slots__ = ("_array",)
    __hash__ = None

    def __init__(self, array: np.ndarray) -> None:
        array = _require_ndarray(array, "ImageArtifact")
        if array.dtype != np.uint8:
            raise ArtifactValidationError(
                f"ImageArtifact requires dtype uint8, got {array.dtype}",
            )
        if array.ndim != 3 or array.shape[2] != 3:
            raise ArtifactValidationError(
                f"ImageArtifact requires shape (height, width, 3), got {array.shape}",
            )
        if array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ArtifactValidationError(
                f"ImageArtifact dimensions must be positive, got {array.shape[:2]}",
            )

        self._array = _owned_c_array(array, np.dtype(np.uint8), writable=True)

    @classmethod
    def from_pil(cls, image: Image.Image) -> ImageArtifact:
        """Create an RGB artifact from a Pillow image without mutating it."""

        if not isinstance(image, Image.Image):
            raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
        converted = image.convert("RGB")
        return cls(np.asarray(converted, dtype=np.uint8))

    @property
    def array(self) -> np.ndarray:
        """Return a writable C-contiguous view of the artifact-owned RGB data."""

        return self._array.view()

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._array.shape

    @property
    def height(self) -> int:
        return int(self._array.shape[0])

    @property
    def width(self) -> int:
        return int(self._array.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self._array.nbytes)

    def copy(self) -> ImageArtifact:
        """Return an independent artifact suitable for another pipeline branch."""

        return type(self)(self._array)

    def to_pil(self) -> Image.Image:
        """Return a detached Pillow RGB image."""

        return Image.fromarray(self._array.copy())

    def __reduce__(self):
        return type(self), (self._array.copy(),)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ImageArtifact) and np.array_equal(self._array, other._array)

    def __repr__(self) -> str:
        return (
            f"ImageArtifact(shape={self.shape!r}, dtype={self._array.dtype}, "
            f"contiguous={self._array.flags.c_contiguous}, writable={self._array.flags.writeable})"
        )


class OriginalWatermark:
    """Immutable one-dimensional binary watermark using ``uint8`` values."""

    contract = DataContract.BINARY_BITS

    __slots__ = ("_values",)
    __hash__ = None

    def __init__(self, values: Iterable[int]) -> None:
        self._values = _normalise_integral_vector(
            values,
            artifact_name="OriginalWatermark",
            dtype=np.dtype(np.uint8),
            allowed_values=frozenset({0, 1}),
        )

    @classmethod
    def from_bit_string(cls, bits: str) -> OriginalWatermark:
        """Create a watermark from a non-empty string containing only 0 and 1."""

        if not isinstance(bits, str):
            raise TypeError(f"bits must be a string, got {type(bits).__name__}")
        if not bits:
            raise ArtifactValidationError("watermark bit string must not be empty")
        invalid = sorted(set(bits) - {"0", "1"})
        if invalid:
            raise ArtifactValidationError(
                "watermark bit string contains invalid characters: " + ", ".join(repr(item) for item in invalid),
            )
        return cls(np.fromiter((int(bit) for bit in bits), dtype=np.uint8, count=len(bits)))

    @property
    def values(self) -> np.ndarray:
        """Return a read-only contiguous vector of binary values."""

        return self._values.view()

    @property
    def length(self) -> int:
        return int(self._values.size)

    def mutable_copy(self, *, dtype: Optional[np.dtype[Any]] = None) -> np.ndarray:
        """Return an independent writable array for a solution entry point."""

        target_dtype = self._values.dtype if dtype is None else np.dtype(dtype)
        return np.array(self._values, dtype=target_dtype, order="C", copy=True)

    def to_bit_string(self) -> str:
        return "".join(str(int(item)) for item in self._values)

    def to_list(self) -> list[int]:
        return [int(item) for item in self._values]

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        return (int(item) for item in self._values)

    def __reduce__(self):
        return type(self), (self._values.copy(),)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OriginalWatermark) and np.array_equal(self._values, other._values)

    def __repr__(self) -> str:
        return f"OriginalWatermark(length={self.length})"


class ExtractedWatermark:
    """Immutable recovered watermark with symbols ``-1``, ``0`` and ``1``.

    ``-1`` represents an erasure or uncertain decision. This class preserves
    erasures instead of silently coercing them to bits; a later metric adapter
    must apply an explicit erasure policy.
    """

    contract = DataContract.TERNARY_BITS

    __slots__ = ("_values",)
    __hash__ = None

    def __init__(self, values: Iterable[int]) -> None:
        self._values = _normalise_integral_vector(
            values,
            artifact_name="ExtractedWatermark",
            dtype=np.dtype(np.int8),
            allowed_values=frozenset({-1, 0, 1}),
        )

    @property
    def values(self) -> np.ndarray:
        """Return a read-only contiguous vector containing -1, 0 and 1."""

        return self._values.view()

    @property
    def length(self) -> int:
        return int(self._values.size)

    @property
    def erasure_count(self) -> int:
        return int(np.count_nonzero(self._values == -1))

    @property
    def has_erasures(self) -> bool:
        return self.erasure_count > 0

    def mutable_copy(self, *, dtype: Optional[np.dtype[Any]] = None) -> np.ndarray:
        """Return an independent writable array for a solution or metric adapter."""

        target_dtype = self._values.dtype if dtype is None else np.dtype(dtype)
        return np.array(self._values, dtype=target_dtype, order="C", copy=True)

    def to_bit_string(self) -> str:
        """Return binary text, failing if an explicit erasure is present."""

        if self.has_erasures:
            raise ArtifactValidationError(
                "cannot convert an extracted watermark containing erasures to a binary string"
            )
        return "".join(str(int(item)) for item in self._values)

    def to_list(self) -> list[int]:
        return [int(item) for item in self._values]

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        return (int(item) for item in self._values)

    def __reduce__(self):
        return type(self), (self._values.copy(),)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExtractedWatermark) and np.array_equal(self._values, other._values)

    def __repr__(self) -> str:
        return f"ExtractedWatermark(length={self.length}, erasures={self.erasure_count})"


__all__ = [
    "ExtractedWatermark",
    "ImageArtifact",
    "OriginalWatermark",
]
