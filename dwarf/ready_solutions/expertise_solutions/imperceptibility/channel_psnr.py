"""Метрика Channel PSNR: качество изображения по одному цветовому каналу."""

import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Imperceptibility_Expertise
from dwarf.ready_solutions.utils.expertise_utils import to_rgb_float


class Channel_PSNR(Ready_Imperceptibility_Expertise):
    """
    Пиковое отношение сигнала к шуму по отдельному каналу RGB.

    Соответствует поканальным значениям PSNR в таблице 1 статьи DCT.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает PSNR между исходным и искажённым каналом изображения.

        Args:
            args (dict): параметры метрики
                original_image (np.ndarray): матрица исходного RGB-изображения
                distorted_image (np.ndarray): матрица искажённого RGB-изображения
                channel (str): канал 'R', 'G' или 'B' (по умолчанию 'R')

        Returns:
            float: PSNR в децибелах, inf при полном совпадении канала

        Raises:
            ValueError: если формы изображений различаются или канал неизвестен
        """
        defaults = {"original_image": None, "distorted_image": None, "channel": "R"}
        args = {**defaults, **args}
        original = to_rgb_float(args["original_image"])
        distorted = to_rgb_float(args["distorted_image"])
        if original.shape != distorted.shape:
            raise ValueError(f"image shapes differ: {original.shape} and {distorted.shape}")
        channel = str(args["channel"]).upper()
        if channel not in ("R", "G", "B"):
            raise ValueError(f"channel must be 'R', 'G' or 'B', got {args['channel']!r}")
        index = "RGB".index(channel)
        mse = np.mean((original[..., index] - distorted[..., index]) ** 2)
        return float("inf") if mse == 0 else float(10 * np.log10(255**2 / mse))
