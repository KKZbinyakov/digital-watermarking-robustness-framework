"""Атака шумом типа "соль и перец": заменяет случайные пиксели чёрным и белым цветом."""

import numpy as np

from dwarf.core.attack_orchestrator.attack_core import Ready_Noise_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_matrix


class Salt_and_Pepper(Ready_Noise_Attacks):
    """
    Атака шумом типа "соль и перец".

    Заменяет случайную долю пикселей на чисто белые (соль) и чисто чёрные
    (перец) значения по всем каналам, поровну разделяя плотность между ними.
    """

    @staticmethod
    def attack(**args):
        """
        Наносит шум "соль и перец" на изображение.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                density (float): суммарная доля повреждённых пикселей, диапазон [0.001, 0.05] (по умолчанию 0.01)
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

        height, width, _ = data.shape
        num_pixels = height * width

        num_salt = int(np.ceil(density * num_pixels * 0.5))
        num_pepper = int(np.ceil(density * num_pixels * 0.5))

        salt_rows = rng.integers(0, height, num_salt)
        salt_cols = rng.integers(0, width, num_salt)
        data[salt_rows, salt_cols, :] = 255

        pepper_rows = rng.integers(0, height, num_pepper)
        pepper_cols = rng.integers(0, width, num_pepper)
        data[pepper_rows, pepper_cols, :] = 0

        return data
