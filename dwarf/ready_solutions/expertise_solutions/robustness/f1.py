"""Метрика F1: гармоническое среднее точности и полноты детектора ЦВЗ."""

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Robustness_Expertise
from dwarf.ready_solutions.utils.expertise_utils import confusion_counts


class F1(Ready_Robustness_Expertise):
    """
    Гармоническое среднее точности и полноты детектора.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает F1-меру бинарного детектора.

        Args:
            args (dict): параметры метрики
                y_true: истинные метки, последовательность из 0 и 1
                y_pred: предсказанные метки, последовательность из 0 и 1

        Returns:
            float: значение F1 в диапазоне от 0 до 1

        Raises:
            ValueError: если формы меток не совпадают или массивы пусты
        """
        defaults = {"y_true": None, "y_pred": None}
        args = {**defaults, **args}
        true_positive, _, false_positive, false_negative = confusion_counts(args["y_true"], args["y_pred"])
        detected = true_positive + false_positive
        actual = true_positive + false_negative
        precision = true_positive / detected if detected else 0.0
        recall = true_positive / actual if actual else 0.0
        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall))
