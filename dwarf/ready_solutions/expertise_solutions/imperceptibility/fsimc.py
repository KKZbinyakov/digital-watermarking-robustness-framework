"""Метрика FSIMc: цветной вариант FSIM с хроматической составляющей."""

import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Imperceptibility_Expertise
from dwarf.ready_solutions.utils.expertise_utils import downsample, fsim_luma_maps, rgb_to_yiq, to_rgb_float


class FSIMc(Ready_Imperceptibility_Expertise):
    """
    Цветной вариант FSIM: к яркостному сходству добавляется хроматическое.

    Нужен для схем, встраивающих ЦВЗ в цветоразностные каналы: обычный FSIM
    считает только яркость и такое встраивание попросту не заметит.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает FSIMc между двумя изображениями.

        Хроматическое сходство возводится в степень 0.03: вклад цвета в
        восприятие искажения заметно меньше вклада яркости, и такой показатель
        задан в эталонной реализации.

        Args:
            args (dict): параметры метрики
                original_image (np.ndarray): матрица оригинального изображения
                distorted_image (np.ndarray): матрица изображения со встроенным ЦВЗ или после атаки

        Returns:
            float: значение FSIMc, единица при полном совпадении
        """
        defaults = {"original_image": None, "distorted_image": None}
        args = {**defaults, **args}
        original_luma, original_i, original_q = rgb_to_yiq(to_rgb_float(args["original_image"]))
        distorted_luma, distorted_i, distorted_q = rgb_to_yiq(to_rgb_float(args["distorted_image"]))

        factor = max(1, int(round(min(original_luma.shape) / 256)))
        original_luma = downsample(original_luma, factor)
        distorted_luma = downsample(distorted_luma, factor)
        original_i = downsample(original_i, factor)
        distorted_i = downsample(distorted_i, factor)
        original_q = downsample(original_q, factor)
        distorted_q = downsample(distorted_q, factor)

        similarity, weights = fsim_luma_maps(original_luma, distorted_luma)

        stabilizer = 200.0
        chroma_exponent = 0.03
        similarity_i = (2 * original_i * distorted_i + stabilizer) / (original_i**2 + distorted_i**2 + stabilizer)
        similarity_q = (2 * original_q * distorted_q + stabilizer) / (original_q**2 + distorted_q**2 + stabilizer)
        chroma = similarity_i * similarity_q
        chroma = np.sign(chroma) * np.abs(chroma) ** chroma_exponent

        return float(np.sum(similarity * chroma * weights) / np.sum(weights))
