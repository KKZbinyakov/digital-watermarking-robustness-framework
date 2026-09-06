"""Контрактные и round-trip тесты всех зарегистрированных embedding-решений."""

import numpy as np
import pytest

from conftest import make_photo

import dwarf.ready_solutions.embedding_solutions
from dwarf import Embedding_Core


EMBEDDINGS = {
    name: cls
    for name, cls in Embedding_Core.get_registered_embeddings().items()
    if not name.startswith("Ready_")
}
NAMES = sorted(EMBEDDINGS)

WATERMARK = np.array([1, 0, 1, 1], dtype=np.uint8)

TOO_LARGE_WATERMARK = np.resize(
    np.array([0, 1], dtype=np.uint8),
    4096,
)


def make_rgb_image(name):
    """Возвращает минимально практичный RGB uint8 ImageArtifact для метода."""
    size = 512 if name == "DFT" else 128
    return np.ascontiguousarray(
        make_photo(size, size, seed=17),
        dtype=np.uint8,
    )

def embedding_class(name):
    """Возвращает реализацию через публичный реестр Embedding_Core."""
    return Embedding_Core.get_embedding_class_by_name(name)

def embed(name, image, watermark=WATERMARK):
    """Встраивает ЦВЗ только на параметрах по умолчанию."""
    return embedding_class(name).embedding(
        input_image=image,
        watermark_bits=watermark,
    )

def extract(name, image, num_bits=None):
    """Извлекает ЦВЗ только на параметрах по умолчанию."""
    if num_bits is None:
        num_bits = WATERMARK.size

    return embedding_class(name).extraction(
        input_image=image,
        num_bits=num_bits,
    )

def assert_image_artifact(image, expected_shape):
    """Проверяет единый контракт ImageArtifact."""
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8
    assert image.shape == expected_shape
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.flags.c_contiguous
    assert np.all((image >= 0) & (image <= 255))

def assert_extracted_watermark(bits, expected_length):
    """Проверяет единый контракт ExtractedWatermark."""
    assert isinstance(bits, np.ndarray)
    assert bits.dtype == np.int8
    assert bits.shape == (expected_length,)
    assert np.all((bits == 0) | (bits == 1))


@pytest.mark.parametrize("name", NAMES)
def test_round_trip_with_defaults(name):
    """embedding -> extraction на умолчаниях должен безошибочно восстановить ЦВЗ."""
    rgb_image = make_rgb_image(name)
    embedded = embed(name, rgb_image)

    assert_image_artifact(embedded, rgb_image.shape)

    extracted = extract(name, embedded)

    assert_extracted_watermark(extracted, WATERMARK.size)
    assert np.array_equal(extracted.astype(np.uint8), WATERMARK), (
        f"{name}: round-trip на параметрах по умолчанию не восстановил исходный ЦВЗ; "
        f"expected={WATERMARK.tolist()}, got={extracted.tolist()}"
    )

@pytest.mark.parametrize("name", NAMES)
def test_round_trip_after_uint8_rounding(name):
    """ЦВЗ должен переживать явное округление/клиппинг изображения до uint8."""
    rgb_image = make_rgb_image(name)
    embedded = embed(name, rgb_image)

    rounded = np.ascontiguousarray(
        np.clip(
            np.rint(embedded.astype(np.float64)),
            0,
            255,
        ).astype(np.uint8)
    )

    assert_image_artifact(rounded, rgb_image.shape)

    extracted = extract(name, rounded)

    assert_extracted_watermark(extracted, WATERMARK.size)
    assert np.array_equal(extracted.astype(np.uint8), WATERMARK), (
        f"{name}: ЦВЗ потерян после округления результата embedding до uint8; "
        f"expected={WATERMARK.tolist()}, got={extracted.tolist()}"
    )

@pytest.mark.parametrize("name", NAMES)
def test_rejects_watermark_over_capacity(name):
    """Встраивание должно явно отказывать, если ЦВЗ превышает ёмкость изображения."""
    rgb_image = make_rgb_image(name)
    with pytest.raises(ValueError):
        embed(name, rgb_image, TOO_LARGE_WATERMARK)

@pytest.mark.parametrize("name", NAMES)
def test_rejects_empty_watermark(name):
    """Пустой OriginalWatermark и запрос extraction нулевой длины недопустимы."""
    rgb_image = make_rgb_image(name)
    empty = np.empty(0, dtype=np.uint8)

    with pytest.raises(ValueError):
        embed(name, rgb_image, empty)

    with pytest.raises(ValueError):
        extract(name, rgb_image, num_bits=0)

@pytest.mark.parametrize("name", NAMES)
def test_does_not_modify_inputs(name):
    """Embedding/extraction не должны изменять переданные массивы на месте."""
    rgb_image = make_rgb_image(name)
    image_before = rgb_image.copy()
    watermark = WATERMARK.copy()
    watermark_before = watermark.copy()

    embedded = embed(name, rgb_image, watermark)

    assert np.array_equal(rgb_image, image_before), (
        f"{name}.embedding изменил input_image на месте"
    )
    assert np.array_equal(watermark, watermark_before), (
        f"{name}.embedding изменил watermark_bits на месте"
    )

    embedded_before = embedded.copy()
    extract(name, embedded)

    assert np.array_equal(embedded, embedded_before), (
        f"{name}.extraction изменил input_image на месте"
    )
