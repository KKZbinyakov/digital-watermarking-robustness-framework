"""Метрика Recall: доля найденных детектором помеченных изображений."""

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Robustness_Expertise
from dwarf.ready_solutions.utils.expertise_utils import confusion_counts


class Recall(Ready_Robustness_Expertise):
    """
    Доля помеченных изображений, на которых детектор нашёл знак.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает полноту бинарного детектора.

        При отсутствии помеченных изображений возвращает 0.0: измерять нечего.

        Args:
            args (dict): параметры метрики
                y_true: истинные метки, последовательность из 0 и 1
                y_pred: предсказанные метки, последовательность из 0 и 1

        Returns:
            float: значение recall в диапазоне от 0 до 1

        Raises:
            ValueError: если формы меток не совпадают или массивы пусты
        """
        defaults = {"y_true": None, "y_pred": None}
        args = {**defaults, **args}
        true_positive, _, _, false_negative = confusion_counts(args["y_true"], args["y_pred"])
        actual = true_positive + false_negative
        return float(true_positive / actual) if actual else 0.0
