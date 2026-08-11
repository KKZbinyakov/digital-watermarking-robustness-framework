"""Атака импульсным шумом: заменяет случайные пиксели случайными значениями."""

import numpy as np

from dwarf.core.attack_orchestrator.attack_core import Ready_Noise_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_matrix


class Impulse(Ready_Noise_Attacks):
    """
    Атака импульсным шумом.

    Заменяет случайно выбранную долю пикселей на случайные значения по всем
    каналам, моделируя битые пиксели или сбои передачи.
    """

    @staticmethod
    def attack(**args):
        """
        Заменяет случайные пиксели изображения случайными значениями.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                density (float): доля повреждённых пикселей, диапазон [0.001, 0.05] (по умолчанию 0.01)
                seed (int): зерно генератора случайных чисел (по умолчанию None)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если density вне диапазона [0.001, 0.05]
        """
        defaults = {"input_image": None, "density": 0.01, "seed": None}
        args = {**defaults, **args}
        input_image = args["input_image"]
        density = float(args["density"])
        seed = args["seed"]

        if not 0.001 <= density <= 0.05:
            raise ValueError(f"density must be in the range [0.001, 0.05], got {density}")

        rng = np.random.default_rng(seed)
        data = to_matrix(input_image)

        height, width, channels = data.shape
        num_pixels = height * width
        num_impulses = int(np.ceil(density * num_pixels))

        rows = rng.integers(0, height, num_impulses)
        cols = rng.integers(0, width, num_impulses)
        values = rng.integers(0, 256, size=(num_impulses, channels), dtype=np.uint8)

        data[rows, cols, :] = values

        return data
