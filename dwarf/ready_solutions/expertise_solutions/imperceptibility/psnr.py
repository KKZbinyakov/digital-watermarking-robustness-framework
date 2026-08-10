"""Метрика PSNR: пиковое отношение сигнала к шуму по каналам RGB."""

import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Imperceptibility_Expertise
from dwarf.ready_solutions.utils.expertise_utils import to_rgb_float


class PSNR(Ready_Imperceptibility_Expertise):
    """
    Пиковое отношение сигнала к шуму между двумя изображениями.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает PSNR между двумя изображениями по каналам RGB.

        Args:
            args (dict): параметры метрики
                original_image (np.ndarray): матрица оригинального изображения
                distorted_image (np.ndarray): матрица изображения со встроенным ЦВЗ или после атаки

        Returns:
            float: PSNR в децибелах, inf при полном совпадении
        """
        defaults = {"original_image": None, "distorted_image": None}
        args = {**defaults, **args}
        original = to_rgb_float(args["original_image"])
        distorted = to_rgb_float(args["distorted_image"])

        mean_squared_error = np.mean((original - distorted) ** 2)
        if mean_squared_error == 0:
            return float("inf")
        return float(10 * np.log10(255**2 / mean_squared_error))
