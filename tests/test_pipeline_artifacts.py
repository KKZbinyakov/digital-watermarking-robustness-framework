"""Tests for canonical image and watermark artifacts."""

import pickle

import numpy as np
import pytest
from PIL import Image

from dwarf.pipeline import (
    ArtifactValidationError,
    DataContract,
    ExtractedWatermark,
    ImageArtifact,
    OriginalWatermark,
)


def test_image_artifact_owns_writable_contiguous_rgb_data():
    base = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    view = base[:, ::-1, :]
    view.setflags(write=False)

    artifact = ImageArtifact(view)

    assert artifact.contract is DataContract.RGB_UINT8
    assert artifact.shape == (5, 7, 3)
    assert artifact.height == 5
    assert artifact.width == 7
    assert artifact.nbytes == 5 * 7 * 3
    assert artifact.array.dtype == np.uint8
    assert artifact.array.flags.c_contiguous
    assert artifact.array.flags.writeable
    assert not artifact.array.flags.owndata
    assert not np.shares_memory(artifact.array, view)
    np.testing.assert_array_equal(artifact.array, view)

    artifact.array[0, 0] = [1, 2, 3]
    assert view[0, 0].tolist() != [1, 2, 3]


def test_image_artifact_rejects_non_array_wrong_dtype_and_wrong_shape():
    with pytest.raises(ArtifactValidationError, match="numpy.ndarray"):
        ImageArtifact([[[0, 0, 0]]])

    with pytest.raises(ArtifactValidationError, match="dtype uint8"):
        ImageArtifact(np.zeros((2, 3, 3), dtype=np.float32))

    for shape in ((2, 3), (2, 3, 1), (2, 3, 4), (0, 3, 3), (2, 0, 3)):
        with pytest.raises(ArtifactValidationError):
            ImageArtifact(np.zeros(shape, dtype=np.uint8))


def test_image_artifact_copy_is_independent():
    artifact = ImageArtifact(np.full((3, 4, 3), 17, dtype=np.uint8))

    clone = artifact.copy()
    clone.array[0, 0] = 255

    assert artifact.array[0, 0].tolist() == [17, 17, 17]
    assert clone.array[0, 0].tolist() == [255, 255, 255]
    assert not np.shares_memory(artifact.array, clone.array)


def test_image_artifact_pillow_round_trip_is_detached():
    source = Image.new("L", (4, 3), color=80)

    artifact = ImageArtifact.from_pil(source)
    restored = artifact.to_pil()

    assert artifact.shape == (3, 4, 3)
    assert restored.mode == "RGB"
    assert restored.size == (4, 3)
    restored.putpixel((0, 0), (1, 2, 3))
    assert artifact.array[0, 0].tolist() == [80, 80, 80]




def test_image_artifact_from_pil_requires_a_pillow_image():
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        ImageArtifact.from_pil(np.zeros((2, 2, 3), dtype=np.uint8))

def test_image_artifact_is_pickle_safe():
    artifact = ImageArtifact(np.arange(24, dtype=np.uint8).reshape(2, 4, 3))

    restored = pickle.loads(pickle.dumps(artifact))

    assert restored == artifact
    assert restored.array.flags.c_contiguous
    assert restored.array.flags.writeable
    assert not np.shares_memory(restored.array, artifact.array)


def test_original_watermark_normalises_integral_iterables():
    watermark = OriginalWatermark(value for value in [True, 0, 1, False])

    assert watermark.contract is DataContract.BINARY_BITS
    assert watermark.values.dtype == np.uint8
    assert watermark.values.flags.c_contiguous
    assert not watermark.values.flags.writeable
    assert watermark.length == 4
    assert len(watermark) == 4
    assert list(watermark) == [1, 0, 1, 0]
    assert watermark.to_list() == [1, 0, 1, 0]
    assert watermark.to_bit_string() == "1010"


def test_original_watermark_from_bit_string():
    watermark = OriginalWatermark.from_bit_string("001101")

    assert watermark.to_list() == [0, 0, 1, 1, 0, 1]


@pytest.mark.parametrize("bits", ["", "10x1", "10 1", 101])
def test_original_watermark_rejects_invalid_bit_strings(bits):
    with pytest.raises((ArtifactValidationError, TypeError)):
        OriginalWatermark.from_bit_string(bits)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0, 2],
        [-1, 0],
        [0.0, 1.0],
        np.array([[0, 1]], dtype=np.uint8),
        np.array(["0", "1"]),
    ],
)
def test_original_watermark_rejects_invalid_vectors(values):
    with pytest.raises(ArtifactValidationError):
        OriginalWatermark(values)


def test_original_watermark_mutable_copy_does_not_break_contract():
    watermark = OriginalWatermark([0, 1, 1])

    with pytest.raises(ValueError):
        watermark.values[0] = 1
    with pytest.raises(ValueError):
        watermark.values.setflags(write=True)

    mutable = watermark.mutable_copy(dtype=np.int32)
    mutable[0] = 1

    assert mutable.dtype == np.int32
    assert mutable.flags.writeable
    assert watermark.to_bit_string() == "011"


def test_extracted_watermark_preserves_erasures():
    watermark = ExtractedWatermark([-1, 0, 1, -1])

    assert watermark.contract is DataContract.TERNARY_BITS
    assert watermark.values.dtype == np.int8
    assert watermark.values.flags.c_contiguous
    assert not watermark.values.flags.writeable
    assert watermark.length == 4
    assert watermark.erasure_count == 2
    assert watermark.has_erasures
    assert watermark.to_list() == [-1, 0, 1, -1]
    with pytest.raises(ValueError):
        watermark.values.setflags(write=True)
    mutable = watermark.mutable_copy(dtype=np.int32)
    mutable[0] = 1
    assert watermark.to_list() == [-1, 0, 1, -1]
    with pytest.raises(ArtifactValidationError, match="erasures"):
        watermark.to_bit_string()


def test_extracted_watermark_without_erasures_converts_to_text():
    watermark = ExtractedWatermark([1, 0, 1, 1])

    assert not watermark.has_erasures
    assert watermark.erasure_count == 0
    assert watermark.to_bit_string() == "1011"


@pytest.mark.parametrize(
    "values",
    [
        [],
        [-2, 0, 1],
        [0, 1, 2],
        [0.0, 1.0],
        np.array([[0, 1]], dtype=np.int8),
    ],
)
def test_extracted_watermark_rejects_invalid_vectors(values):
    with pytest.raises(ArtifactValidationError):
        ExtractedWatermark(values)


def test_watermark_artifacts_are_pickle_safe():
    original = OriginalWatermark([0, 1, 1, 0])
    extracted = ExtractedWatermark([-1, 1, 0, 0])

    restored_original = pickle.loads(pickle.dumps(original))
    restored_extracted = pickle.loads(pickle.dumps(extracted))

    assert restored_original == original
    assert restored_extracted == extracted
    assert not restored_original.values.flags.writeable
    assert not restored_extracted.values.flags.writeable
