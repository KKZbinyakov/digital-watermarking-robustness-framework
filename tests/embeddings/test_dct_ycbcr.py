"""Проверки DCT Xiong (2016) с прямым входом и выходом YCbCr float64."""

import numpy as np
import pytest
from dwarf.ready_solutions.embedding_solutions.frequency.dct import DCT


def make_ycbcr(height=19, width=27):
    rng = np.random.default_rng(2016)
    image = np.empty((height, width, 3), dtype=np.float64)
    image[:, :, 0] = rng.uniform(0, 255, (height, width))
    image[:, :, 1:] = rng.uniform(-100, 100, (height, width, 2))
    return image


@pytest.mark.parametrize("count", [1, 3, 4, 6])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly"])
def test_round_trip_preserves_inputs_chroma_and_unused_pixels(count, layout):
    image = make_ycbcr()
    if layout == "strided":
        image = image[:, ::-1, :]
    elif layout == "readonly":
        image.flags.writeable = False
    before = image.copy()
    watermark = np.resize(np.array([0, 1], dtype=np.uint8), count)
    watermark_before = watermark.copy()

    result = DCT.embedding(input_image=image, watermark_bits=watermark)

    assert result.shape == image.shape
    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, image)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(watermark, watermark_before)
    np.testing.assert_array_equal(result[:, :, 1:], before[:, :, 1:])
    used = np.zeros(image.shape[:2], dtype=bool)
    blocks_w = image.shape[1] // 8
    for index in range(count):
        row, col = divmod(index, blocks_w)
        used[row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8] = True
    np.testing.assert_array_equal(result[~used], before[~used])
    assert np.any(result[:, :, 0][used] != before[:, :, 0][used])

    result_before = result.copy()
    if layout == "readonly":
        result.flags.writeable = False
    elif layout == "strided":
        storage = np.empty((result.shape[0], result.shape[1] * 2, 3))
        storage[:, ::2] = result
        result = storage[:, ::2]
    extracted = DCT.extraction(input_image=result, num_bits=count)
    assert extracted.dtype == np.int8
    np.testing.assert_array_equal(extracted, watermark)
    np.testing.assert_array_equal(result, result_before)


def test_embedding_and_extraction_use_only_y():
    first = make_ycbcr()
    second = first.copy()
    second[:, :, 1:] *= -0.5
    bits = np.array([1, 0, 1, 0], dtype=np.uint8)
    a = DCT.embedding(input_image=first, watermark_bits=bits)
    b = DCT.embedding(input_image=second, watermark_bits=bits)
    np.testing.assert_array_equal(a[:, :, 0], b[:, :, 0])
    a[:, :, 1:] = second[:, :, 1:]
    np.testing.assert_array_equal(DCT.extraction(input_image=a, num_bits=4), bits)


@pytest.mark.parametrize("luma", [0.0, 255.0])
def test_no_clipping_or_uint8_rounding(luma):
    image = np.zeros((8, 8, 3), dtype=np.float64)
    image[:, :, 0] = luma
    image[:, :, 1:] = [-31.25, 42.125]
    bits = np.array([1], dtype=np.uint8)
    result = DCT.embedding(input_image=image, watermark_bits=bits)
    assert np.any(result[:, :, 0] != np.rint(result[:, :, 0]))
    assert result[:, :, 0].min() < 0 if luma == 0 else result[:, :, 0].max() > 255
    np.testing.assert_array_equal(result[:, :, 1:], image[:, :, 1:])
    np.testing.assert_array_equal(DCT.extraction(input_image=result, num_bits=1), bits)


@pytest.mark.parametrize("method", ["embedding", "extraction"])
@pytest.mark.parametrize("dtype", [np.uint8, np.int32, np.float32])
def test_rejects_old_uint8_contract_and_other_dtypes(method, dtype):
    kwargs = {"watermark_bits": np.array([1], dtype=np.uint8)} if method == "embedding" else {"num_bits": 1}
    with pytest.raises(TypeError, match="YCbCr with dtype float64"):
        getattr(DCT, method)(input_image=np.zeros((8, 8, 3), dtype=dtype), **kwargs)


@pytest.mark.parametrize("method", ["embedding", "extraction"])
@pytest.mark.parametrize("shape", [(8, 8), (8, 8, 4), (7, 8, 3), (8, 7, 3)])
def test_rejects_invalid_shape(method, shape):
    kwargs = {"watermark_bits": np.array([1], dtype=np.uint8)} if method == "embedding" else {"num_bits": 1}
    with pytest.raises(ValueError):
        getattr(DCT, method)(input_image=np.zeros(shape, dtype=np.float64), **kwargs)


@pytest.mark.parametrize("method", ["embedding", "extraction"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("channel", [0, 1, 2])
def test_rejects_nonfinite_channels(method, value, channel):
    image = make_ycbcr()
    image[0, 0, channel] = value
    kwargs = {"watermark_bits": np.array([1], dtype=np.uint8)} if method == "embedding" else {"num_bits": 1}
    with pytest.raises(ValueError, match="finite YCbCr"):
        getattr(DCT, method)(input_image=image, **kwargs)


@pytest.mark.parametrize("count", [0, 7])
def test_rejects_empty_or_over_capacity_requests(count):
    image = make_ycbcr()
    with pytest.raises(ValueError):
        DCT.embedding(input_image=image, watermark_bits=np.zeros(count, dtype=np.uint8))
    with pytest.raises(ValueError):
        DCT.extraction(input_image=image, num_bits=count)


@pytest.mark.parametrize("strength", [0, -1, np.nan, np.inf])
def test_rejects_invalid_strength(strength):
    with pytest.raises(ValueError, match="strength"):
        DCT.embedding(input_image=make_ycbcr(), watermark_bits=np.array([1], dtype=np.uint8), strength=strength)
