"""Метрика FSIM: индекс сходства признаков по яркостной составляющей."""

import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Imperceptibility_Expertise
from dwarf.ready_solutions.utils.expertise_utils import downsample, fsim_luma_maps, to_gray


class FSIM(Ready_Imperceptibility_Expertise):
    """
    Feature Similarity Index по яркостной составляющей.

    Опирается на фазовую согласованность как меру воспринимаемой значимости
    точки и на модуль градиента Шарра как меру контраста. Фазовая
    согласованность инвариантна к яркости и контрасту, поэтому метрика
    устойчива к равномерным сдвигам тона, которые PSNR штрафует.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает FSIM между двумя изображениями.

        Перед расчётом кадр прореживается до примерно 256 пикселей по меньшей
        стороне, как в эталонной реализации: банк фильтров log-Gabor настроен на
        этот масштаб, и без прореживания значения на крупных кадрах смещаются.

        Args:
            args (dict): параметры метрики
                original_image (np.ndarray): матрица оригинального изображения
                distorted_image (np.ndarray): матрица изображения со встроенным ЦВЗ или после атаки

        Returns:
            float: значение FSIM, единица при полном совпадении
        """
        defaults = {"original_image": None, "distorted_image": None}
        args = {**defaults, **args}
        original_luma = to_gray(args["original_image"])
        distorted_luma = to_gray(args["distorted_image"])

        factor = max(1, int(round(min(original_luma.shape) / 256)))
        original_luma = downsample(original_luma, factor)
        distorted_luma = downsample(distorted_luma, factor)

        similarity, weights = fsim_luma_maps(original_luma, distorted_luma)
        return float(np.sum(similarity * weights) / np.sum(weights))
