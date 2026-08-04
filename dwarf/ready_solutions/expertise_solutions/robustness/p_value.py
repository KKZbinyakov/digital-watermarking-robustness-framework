"""Метрика P-Value: эмпирическое одностороннее p-значение детектора ЦВЗ."""

import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Robustness_Expertise


class P_Value(Ready_Robustness_Expertise):
    """
    Эмпирическое одностороннее p-значение для статистических детекторов.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает долю выборок нулевого распределения, не меньших наблюдаемой статистики.

        В числитель и знаменатель добавляется единица: без этого p-значение может
        оказаться ровно нулевым, что утверждает невозможность события по конечной
        выборке. С поправкой минимум равен 1 / (N + 1) и честно отражает предел
        разрешения выборки.

        Args:
            args (dict): параметры метрики
                statistic (float): наблюдаемое значение статистики детектора
                null_samples: выборка значений статистики на изображениях без знака

        Returns:
            float: p-значение в диапазоне от 0 до 1, nan при пустой выборке
        """
        defaults = {"statistic": None, "null_samples": None}
        args = {**defaults, **args}
        null_samples = np.asarray(args["null_samples"], dtype=float)
        if null_samples.size == 0:
            return float("nan")
        statistic = float(args["statistic"])
        return float((np.sum(null_samples >= statistic) + 1) / (null_samples.size + 1))
