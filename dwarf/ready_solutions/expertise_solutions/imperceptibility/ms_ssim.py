"""Метрика MS-SSIM: мультимасштабный индекс структурного сходства."""

import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Imperceptibility_Expertise
from dwarf.ready_solutions.utils.expertise_utils import downsample_by_two, gauss1d, ssim_maps, to_gray


class MS_SSIM(Ready_Imperceptibility_Expertise):
    """
    Мультимасштабный SSIM: пять масштабов со стандартными весами.

    Обычный SSIM привязан к размеру окна и потому к разрешению кадра.
    Мультимасштабный вариант снимает эту зависимость, оценивая структуру на
    последовательно прореженных копиях, и лучше согласуется с восприятием.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает MS-SSIM между двумя изображениями.

        Args:
            args (dict): параметры метрики
                original_image (np.ndarray): матрица оригинального изображения
                distorted_image (np.ndarray): матрица изображения со встроенным ЦВЗ или после атаки

        Returns:
            float: значение MS-SSIM, единица при полном совпадении

        Raises:
            ValueError: если кадр меньше 176 пикселей по любой стороне
        """
        defaults = {"original_image": None, "distorted_image": None}
        args = {**defaults, **args}
        original = to_gray(args["original_image"])
        distorted = to_gray(args["distorted_image"])

        scales = 5
        window = gauss1d(11, 1.5)
        minimum_side = 2 ** (scales - 1) * window.shape[0]
        if min(original.shape) < minimum_side:
            raise ValueError(
                f"MS-SSIM requires an image of at least {minimum_side} pixels on each side, got {original.shape}"
            )

        weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

        scale_ssim = []
        scale_cs = []
        for _ in range(scales):
            ssim_map, cs_map = ssim_maps(original, distorted, window)
            scale_ssim.append(ssim_map.mean())
            scale_cs.append(cs_map.mean())
            original = downsample_by_two(original)
            distorted = downsample_by_two(distorted)

        scale_cs = np.clip(np.array(scale_cs), 1e-8, None)
        scale_ssim = np.clip(np.array(scale_ssim), 1e-8, None)
        return float(np.prod(scale_cs[:-1] ** weights[:-1]) * (scale_ssim[-1] ** weights[-1]))
