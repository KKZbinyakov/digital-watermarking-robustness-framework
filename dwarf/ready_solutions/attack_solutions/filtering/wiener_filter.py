"""Атака винеровской фильтрацией: адаптивно сглаживает изображение с учётом локальной дисперсии."""

import numpy as np
from scipy.signal import wiener

from dwarf.core.attack_orchestrator.attack_core import Ready_Filtering_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_matrix


class Wiener_Filter(Ready_Filtering_Attacks):
    """
    Атака винеровской фильтрацией.

    Применяет к каждому каналу изображения адаптивный винеровский фильтр,
    сглаживающий изображение с учётом локальной дисперсии и оценки шума,
    что может подавлять слабый сигнал водяного знака.
    """

    @staticmethod
    def attack(**args):
        """
        Применяет винеровский фильтр к каждому каналу изображения.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                window (int): размер окна фильтра, нечётное число >= 3 (по умолчанию 5)
                noise (float): оценка дисперсии шума, при 0.0 оценивается автоматически по локальной дисперсии (по умолчанию 0.0)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если window меньше 3 или чётное, либо noise отрицательный
        """
        defaults = {"input_image": None, "window": 5, "noise": 0.0}
        args = {**defaults, **args}
        input_image = args["input_image"]
        window = args["window"]
        noise = float(args["noise"])

        if window < 3 or window % 2 == 0:
            raise ValueError(f"window must be an odd number >= 3, got {window}")
        if noise < 0:
            raise ValueError(f"noise must not be negative, got {noise}")

        data = to_matrix(input_image).astype(np.float32) / 255.0
        wiener_noise = None if noise == 0.0 else noise

        filtered = np.empty_like(data)
        for c in range(data.shape[2]):
            filtered[..., c] = wiener(data[..., c], mysize=(window, window), noise=wiener_noise)

        filtered = np.nan_to_num(filtered, nan=0.0)
        return to_matrix(np.clip(filtered, 0, 1) * 255.0)
