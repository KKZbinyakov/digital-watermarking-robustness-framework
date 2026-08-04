"""Метрика AUC: площадь под ROC-кривой детектора ЦВЗ."""

import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Robustness_Expertise
from dwarf.ready_solutions.utils.expertise_utils import avg_ranks


class AUC(Ready_Robustness_Expertise):
    """
    Площадь под ROC-кривой детектора наличия ЦВЗ.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает AUC через статистику Манна-Уитни.

        Ранговая формула даёт тот же результат, что интегрирование ROC-кривой,
        но не требует перебора порогов и корректно обрабатывает совпадающие оценки.

        Args:
            args (dict): параметры метрики
                y_true: истинные метки, последовательность из 0 и 1
                y_scores: непрерывные оценки детектора той же длины

        Returns:
            float: значение AUC в диапазоне от 0 до 1, nan если один из классов пуст

        Raises:
            ValueError: если длины меток и оценок не совпадают
        """
        defaults = {"y_true": None, "y_scores": None}
        args = {**defaults, **args}
        y_true = np.asarray(args["y_true"])
        y_scores = np.asarray(args["y_scores"], dtype=float)
        if y_true.shape != y_scores.shape:
            raise ValueError(f"shapes differ: labels {y_true.shape}, scores {y_scores.shape}")

        positives = int((y_true == 1).sum())
        negatives = int((y_true == 0).sum())
        if positives == 0 or negatives == 0:
            return float("nan")

        ranks = avg_ranks(y_scores)
        return float((ranks[y_true == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
