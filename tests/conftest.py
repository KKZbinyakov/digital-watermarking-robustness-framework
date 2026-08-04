"""Общие фикстуры и помощники тестов.

Все решения фреймворка работают с матрицами изображений, поэтому фикстуры
отдают именно матрицы: ни один тест не пишет файлы на диск и не открывает их.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def make_photo(height=288, width=288, seed=20240501):
    """
    Строит воспроизводимую матрицу, похожую на фотографию.

    Чистый шум и чистый градиент — вырожденные случаи: на шуме вырождается
    фазовая согласованность, на градиенте — гистограммные атаки. Здесь есть и
    плавные области, и резкие границы, и мелкая фактура.

    Args:
        height (int): высота кадра
        width (int): ширина кадра
        seed (int): зерно генератора

    Returns:
        np.ndarray: матрица формы (height, width, 3) типа uint8
    """
    rng = np.random.default_rng(seed)
    rows = np.linspace(0, 1, height)[:, None]
    columns = np.linspace(0, 1, width)[None, :]

    background = 80 + 90 * np.sin(6 * rows) * np.cos(5 * columns)
    disc = ((rows - 0.35) ** 2 + (columns - 0.6) ** 2) < 0.03
    stripes = ((np.arange(width)[None, :] // 8) % 2).astype(np.float64) * 45
    texture = rng.normal(0.0, 6.0, (height, width))

    luma = background + stripes + texture
    luma[disc] += 70

    image = np.empty((height, width, 3), dtype=np.float64)
    image[:, :, 0] = luma * 1.05
    image[:, :, 1] = luma
    image[:, :, 2] = luma * 0.92 + 20
    return np.clip(image, 0, 255).round().astype(np.uint8)


@pytest.fixture
def photo():
    """Матрица изображения, достаточно большая для MS-SSIM и VIF."""
    return make_photo()


@pytest.fixture
def small_photo():
    """Небольшая матрица для тестов, где размер не важен."""
    return make_photo(64, 96, seed=7)


@pytest.fixture
def detector():
    """
    Метки и оценки бинарного детектора наличия ЦВЗ.

    Returns:
        dict: y_true, y_pred и y_scores одинаковой длины
    """
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    y_scores = np.array([0.9, 0.1, 0.8, 0.35, 0.2, 0.7, 0.6, 0.3, 0.85, 0.4])
    return {"y_true": y_true, "y_pred": y_pred, "y_scores": y_scores}


def solutions(registry):
    """
    Оставляет только конкретные реализации, отбрасывая категории Ready_*.

    Args:
        registry (dict): реестр решений вида имя -> класс

    Returns:
        dict: реестр без абстрактных категорий
    """
    return {name: cls for name, cls in registry.items() if not name.startswith("Ready_")}
