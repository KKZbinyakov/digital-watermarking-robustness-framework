"""Атака гамма-коррекции: поканальное степенное преобразование уровней."""

import numpy as np

from dwarf.core.attack_orchestrator.attack_core import Ready_Color_Brightness_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_matrix


class Gamma_Correction(Ready_Color_Brightness_Attacks):
    """
    Атака гамма-коррекции.

    Применяет поканальное степенное преобразование: значения gamma меньше
    единицы осветляют средние тона, больше единицы - затемняют
    """

    @staticmethod
    def attack(**args):
        """
        Применяет гамма-коррекцию к изображению.

        Результат зависит только от исходного уровня пикселя, поэтому
        преобразование считается один раз на таблицу из 256 значений.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                gamma (float): коэффициент гаммы, больше нуля (по умолчанию 1.5)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если gamma не положительна
        """
        defaults = {"input_image": None, "gamma": 1.5}
        args = {**defaults, **args}
        input_image = args["input_image"]
        gamma = float(args["gamma"])

        if gamma <= 0:
            raise ValueError(f"gamma must be greater than zero, got {gamma}")

        levels = np.arange(256, dtype=np.float64) / 255.0
        lookup = np.clip(255.0 * levels**gamma, 0, 255).round().astype(np.uint8)
        return lookup[to_matrix(input_image)]
