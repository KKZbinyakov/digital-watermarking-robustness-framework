"""Атака мультипликативным (спекл) шумом: добавляет шум, пропорциональный яркости пикселя."""

import numpy as np

from dwarf.core.attack_orchestrator.attack_core import Ready_Noise_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_matrix


class Speckle(Ready_Noise_Attacks):
    """
    Атака мультипликативным (спекл) шумом.

    Добавляет к пикселю шум, пропорциональный его собственной яркости
    (pixel + pixel * noise), моделируя спекл-шум когерентных изображающих систем.
    """

    @staticmethod
    def attack(**args):
        """
        Добавляет мультипликативный шум к изображению.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                variance (float): дисперсия шума, диапазон [0.001, 0.05] (по умолчанию 0.05)
                seed (int): зерно генератора случайных чисел (по умолчанию None)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если variance вне диапазона [0.001, 0.05]
        """
        defaults = {"input_image": None, "variance": 0.05, "seed": None}
        args = {**defaults, **args}
        input_image = args["input_image"]
        variance = float(args["variance"])
        seed = args["seed"]

        if not 0.001 <= variance <= 0.05:
            raise ValueError(f"variance must be in the range [0.001, 0.05], got {variance}")

        rng = np.random.default_rng(seed)
        data = to_matrix(input_image).astype(np.float32) / 255.0

        sigma = np.sqrt(variance)
        noise = rng.normal(0.0, sigma, data.shape).astype(np.float32)

        noisy = np.clip(data + data * noise, 0, 1) * 255.0
        return to_matrix(noisy)
