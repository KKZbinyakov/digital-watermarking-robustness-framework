"""Общие вспомогательные функции атак."""

import io

import numpy as np
from PIL import Image


def to_matrix(image) -> np.ndarray:
    """
    Приводит вход атаки к матрице целых уровней RGB.

    Уровни округляются и прижимаются к диапазону 0..255: атаки считают
    промежуточные результаты в вещественных числах, а на границах контракта
    изображение всегда целочисленное.

    Результат создаётся копированием, поэтому атака может менять его на месте,
    не затрагивая матрицу вызывающего кода.

    Args:
        image (np.ndarray): матрица формы (H, W, 3) любого числового типа

    Returns:
        np.ndarray: матрица формы (H, W, 3) типа uint8

    Raises:
        ValueError: если матрица не имеет формы (H, W, 3)
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an RGB image matrix of shape (H, W, 3), got {array.shape}")
    return np.clip(array, 0, 255).round().astype(np.uint8)


def to_pil(image) -> Image.Image:
    """
    Приводит матрицу изображения к трёхканальному изображению Pillow.

    Args:
        image (np.ndarray): матрица формы (H, W, 3) любого числового типа

    Returns:
        Image.Image: изображение в режиме 'RGB'

    Raises:
        ValueError: если матрица не имеет формы (H, W, 3)
    """
    return Image.fromarray(to_matrix(image), "RGB")


def to_array(img: Image.Image) -> np.ndarray:
    """
    Приводит изображение Pillow к матрице уровней RGB.

    Массив создаётся копированием, поэтому его можно менять на месте: атаки
    возвращают результат вызывающему коду и не должны разделять с ним память.

    Args:
        img (Image.Image): изображение в любом режиме

    Returns:
        np.ndarray: матрица формы (H, W, 3) типа uint8
    """
    return np.array(img.convert("RGB"))


def load_rgb(image_path: str) -> np.ndarray:
    """
    Открывает изображение и приводит его к трёхканальному RGB.

    Args:
        image_path (str): путь к изображению

    Returns:
        np.ndarray: массив формы (H, W, 3) типа uint8
    """
    return to_array(Image.open(image_path))


def save_rgb(array: np.ndarray, output_path: str) -> None:
    """
    Сохраняет массив как RGB-изображение с округлением до целых уровней.

    Args:
        array (np.ndarray): массив формы (H, W, 3) любого числового типа
        output_path (str): путь для сохранения результата

    Returns:
        None
    """
    to_pil(array).save(output_path)


def roundtrip_buffer(image, fmt: str, **save_kwargs) -> np.ndarray:
    """
    Прогоняет изображение через кодек в памяти: кодирует и сразу декодирует.

    Файл на диске не создаётся, поэтому искажение кодека отделено от искажений
    контейнера, в котором результат мог бы сохраняться дальше.

    Args:
        image (np.ndarray): матрица изображения формы (H, W, 3)
        fmt (str): имя формата Pillow, например 'JPEG' или 'WEBP'
        save_kwargs: параметры кодека, передаваемые в Image.save

    Returns:
        np.ndarray: матрица изображения после кодирования и декодирования
    """
    buffer = io.BytesIO()
    to_pil(image).save(buffer, format=fmt, **save_kwargs)
    buffer.seek(0)
    return to_array(Image.open(buffer))
