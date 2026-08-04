import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Imperceptibility_Expertise
from dwarf.ready_solutions.utils.expertise_utils import downsample, fsim_luma_maps, load_rgb_float, rgb_to_yiq


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
                original_path (str): путь к оригинальному изображению
                distorted_path (str): путь к изображению со встроенным ЦВЗ или после атаки

        Returns:
            float: значение FSIM, единица при полном совпадении
        """
        defaults = {} # Написать дефолтные значения
        args = {**defaults, **args}
        original_luma, _, _ = rgb_to_yiq(load_rgb_float(args["original_path"]))
        distorted_luma, _, _ = rgb_to_yiq(load_rgb_float(args["distorted_path"]))

        factor = max(1, int(round(min(original_luma.shape) / 256)))
        original_luma = downsample(original_luma, factor)
        distorted_luma = downsample(distorted_luma, factor)

        similarity, weights = fsim_luma_maps(original_luma, distorted_luma)
        return float(np.sum(similarity * weights) / np.sum(weights))
