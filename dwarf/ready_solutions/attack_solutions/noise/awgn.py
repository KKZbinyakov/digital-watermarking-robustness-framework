"""Атака аддитивным белым гауссовым шумом (AWGN): добавляет гауссов шум к каждому пикселю."""

import numpy as np

from dwarf.core.attack_orchestrator.attack_core import Ready_Noise_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_matrix


class AWGN(Ready_Noise_Attacks):
    """
    Атака аддитивным белым гауссовым шумом (AWGN).

    Добавляет к каждому пикселю независимый гауссов шум с нулевым средним,
    моделируя шум датчика или канала передачи.
    """

    @staticmethod
    def attack(**args):
        """
        Добавляет гауссов шум к изображению.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                sigma (float): стандартное отклонение шума в долях канала 0..1, диапазон [0.001, 0.1] (по умолчанию 0.01)
                seed (int): зерно генератора случайных чисел (по умолчанию None)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если sigma вне диапазона [0.001, 0.1]
        """
        defaults = {"input_image": None, "sigma": 0.01, "seed": None}
        args = {**defaults, **args}
        input_image = args["input_image"]
        sigma = float(args["sigma"])
        seed = args["seed"]

        if not 0.001 <= sigma <= 0.1:
            raise ValueError(f"sigma must be in the range [0.001, 0.1], got {sigma}")

        rng = np.random.default_rng(seed)
        data = to_matrix(input_image).astype(np.float32) / 255.0

        noise = rng.normal(0, sigma, data.shape)
        noisy = np.clip(data + noise, 0, 1) * 255.0
        return to_matrix(noisy)
