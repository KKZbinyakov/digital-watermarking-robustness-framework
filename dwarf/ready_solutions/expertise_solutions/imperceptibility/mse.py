"""Метрика MSE: среднеквадратичная ошибка по каналам RGB."""

import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Imperceptibility_Expertise
from dwarf.ready_solutions.utils.expertise_utils import to_rgb_float


class MSE(Ready_Imperceptibility_Expertise):
    """
    Среднеквадратичная ошибка по каналам RGB.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает среднеквадратичную ошибку между двумя изображениями.

        Args:
            args (dict): параметры метрики
                original_image (np.ndarray): матрица оригинального изображения
                distorted_image (np.ndarray): матрица изображения со встроенным ЦВЗ или после атаки

        Returns:
            float: значение MSE, не меньше нуля
        """
        defaults = {"original_image": None, "distorted_image": None}
        args = {**defaults, **args}
        original = to_rgb_float(args["original_image"])
        distorted = to_rgb_float(args["distorted_image"])

        return float(np.mean((original - distorted) ** 2))
