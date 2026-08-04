"""Метрика Watermark PSNR: качество восстановления изображения-знака."""

import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Robustness_Expertise
from dwarf.ready_solutions.utils.expertise_utils import to_gray


class Watermark_PSNR(Ready_Robustness_Expertise):
    """
    PSNR между исходным и восстановленным изображением-знаком.

    Применяется к схемам, где ЦВЗ представляет собой картинку, а не битовую
    строку: там BER неприменим, а качество восстановления всё равно надо мерить.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает PSNR между двумя изображениями-знаками по каналу яркости.

        Изображения обрезаются до общего размера, если он различается.

        Args:
            args (dict): параметры метрики
                original_watermark (np.ndarray): матрица исходного изображения-знака
                extracted_watermark (np.ndarray): матрица восстановленного изображения-знака

        Returns:
            float: PSNR в децибелах, inf при полном совпадении
        """
        defaults = {"original_watermark": None, "extracted_watermark": None}
        args = {**defaults, **args}
        original = to_gray(args["original_watermark"])
        extracted = to_gray(args["extracted_watermark"])

        rows = min(original.shape[0], extracted.shape[0])
        cols = min(original.shape[1], extracted.shape[1])
        original = original[:rows, :cols]
        extracted = extracted[:rows, :cols]

        mean_squared_error = np.mean((original - extracted) ** 2)
        if mean_squared_error == 0:
            return float("inf")
        return float(10 * np.log10(255**2 / mean_squared_error))
