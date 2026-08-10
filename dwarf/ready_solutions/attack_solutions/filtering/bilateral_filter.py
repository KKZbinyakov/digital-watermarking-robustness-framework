"""Атака билатеральной фильтрацией: сглаживает изображение с учётом пространственной и цветовой близости."""

import numpy as np

from dwarf.core.attack_orchestrator.attack_core import Ready_Filtering_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_matrix


class Bilateral_Filter(Ready_Filtering_Attacks):
    """
    Атака билатеральной фильтрацией.

    Сглаживает изображение с учётом как пространственной близости, так и
    близости по цвету, что позволяет размывать шум и водяной знак, сохраняя
    резкие границы.

    Требует пакет opencv-python-headless
    """

    @staticmethod
    def attack(**args):
        """
        Применяет билатеральный фильтр к изображению.

        Диаметр окна фильтра вычисляется из sigma_space, чтобы захватывать
        значимую часть гауссова пространственного ядра.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                sigma_color (float): сила сглаживания по цвету в долях канала 0..1, диапазон [0.01, 0.5] (по умолчанию 0.1)
                sigma_space (float): сила пространственного сглаживания в пикселях, диапазон [1.0, 10.0] (по умолчанию 3.0)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если sigma_color вне диапазона [0.01, 0.5] или sigma_space вне диапазона [1.0, 10.0]
            RuntimeError: если пакет opencv-python-headless не установлен
        """
        defaults = {"input_image": None, "sigma_color": 0.1, "sigma_space": 3.0}
        args = {**defaults, **args}
        input_image = args["input_image"]
        sigma_color = float(args["sigma_color"])
        sigma_space = float(args["sigma_space"])

        if not 0.01 <= sigma_color <= 0.5:
            raise ValueError(f"sigma_color must be in the range [0.01, 0.5], got {sigma_color}")
        if not 1.0 <= sigma_space <= 10.0:
            raise ValueError(f"sigma_space must be in the range [1.0, 10.0], got {sigma_space}")

        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "Bilateral_Filter attack requires the opencv-python-headless package: "
                "pip install opencv-python-headless"
            ) from error

        data = to_matrix(input_image).astype(np.float32) / 255.0

        d = int(2 * np.ceil(2 * sigma_space) + 1)  # Диаметр окна

        filtered = cv2.bilateralFilter(data, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        return to_matrix(np.clip(filtered, 0, 1) * 255.0)
