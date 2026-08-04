import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Imperceptibility_Expertise
from dwarf.ready_solutions.utils.expertise_utils import load_rgb_float


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
                original_path (str): путь к оригинальному изображению
                distorted_path (str): путь к изображению со встроенным ЦВЗ или после атаки

        Returns:
            float: значение MSE, не меньше нуля
        """
        defaults = {} # Написать дефолтные значения
        args = {**defaults, **args}
        original = load_rgb_float(args["original_path"])
        distorted = load_rgb_float(args["distorted_path"])
        return float(np.mean((original - distorted) ** 2))
