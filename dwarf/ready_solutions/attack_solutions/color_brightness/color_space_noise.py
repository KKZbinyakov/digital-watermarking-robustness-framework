"""Атака гауссовым шумом в заданном цветовом пространстве."""

import numpy as np
from PIL import Image

from dwarf.core.attack_orchestrator.attack_core import Ready_Color_Brightness_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_array, to_pil


class Color_Space_Noise(Ready_Color_Brightness_Attacks):
    """
    Атака гауссовым шумом в заданном цветовом пространстве.
    """

    @staticmethod
    def attack(**args):
        """
        Добавляет шум в заданном цветовом пространстве.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                space (str): цветовое пространство Pillow, 'YCbCr', 'HSV' или 'LAB' (по умолчанию 'YCbCr')
                noise_std (float): стандартное отклонение шума в единицах канала 0..255 (по умолчанию 10.0)
                seed (int): зерно генератора случайных чисел (по умолчанию None)

        Returns:
            np.ndarray: матрица изображения после атаки
        """
        defaults = {"input_image": None, "space": "YCbCr", "noise_std": 10.0, "seed": None}
        args = {**defaults, **args}
        input_image = args["input_image"]
        space = args["space"]
        noise_std = float(args["noise_std"])
        seed = args["seed"]

        rng = np.random.default_rng(seed)
        converted = np.asarray(to_pil(input_image).convert(space)).astype(np.float64)
        converted += rng.normal(0.0, noise_std, converted.shape)
        converted = np.clip(converted, 0, 255).round().astype(np.uint8)
        return to_array(Image.fromarray(converted, space))
