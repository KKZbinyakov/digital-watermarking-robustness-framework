"""Атака пуассоновским (дробовым) шумом: моделирует шум фотонного дробового эффекта."""

import numpy as np

from dwarf.core.attack_orchestrator.attack_core import Ready_Noise_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_matrix


class Poisson(Ready_Noise_Attacks):
    """
    Атака пуассоновским (дробовым) шумом.

    Моделирует шум фотонного дробового эффекта: масштабирует яркость пикселя
    к среднему числу фотонов peak, сэмплирует пуассоновскую случайную величину
    и масштабирует обратно.
    """

    @staticmethod
    def attack(**args):
        """
        Добавляет пуассоновский шум к изображению.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                peak (float): среднее число фотонов на максимум яркости, диапазон [1.0, 1000.0] (по умолчанию 30.0)
                seed (int): зерно генератора случайных чисел (по умолчанию None)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если peak вне диапазона [1.0, 1000.0]
        """
        defaults = {"input_image": None, "peak": 30.0, "seed": None}
        args = {**defaults, **args}
        input_image = args["input_image"]
        peak = float(args["peak"])
        seed = args["seed"]

        if not 1.0 <= peak <= 1000.0:
            raise ValueError(f"peak must be in the range [1.0, 1000.0], got {peak}")

        rng = np.random.default_rng(seed)
        data = to_matrix(input_image).astype(np.float32) / 255.0

        scaled = data * peak
        noisy = rng.poisson(scaled).astype(np.float32) / peak

        noisy = np.clip(noisy, 0, 1) * 255.0
        return to_matrix(noisy)
