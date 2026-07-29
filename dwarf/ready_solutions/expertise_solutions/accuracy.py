from ..utils.expertise_utils import *


class Accuracy(Ready_Robustness_Expertise):
    """
    Доля верных решений бинарного детектора наличия ЦВЗ.
    """

    @staticmethod
    def expertise(args: dict = {
        "y_true": None,
        "y_pred": None
    }):
        """
        Считает долю совпадений предсказанных меток с истинными.

        Args:
            args (dict): параметры метрики
                y_true: истинные метки, последовательность из 0 и 1
                y_pred: предсказанные метки, последовательность из 0 и 1

        Returns:
            float: значение accuracy в диапазоне от 0 до 1

        Raises:
            ValueError: если формы меток не совпадают или массивы пусты
        """
        true_positive, true_negative, false_positive, false_negative = confusion_counts(
            args["y_true"], args["y_pred"]
        )
        total = true_positive + true_negative + false_positive + false_negative
        return float((true_positive + true_negative) / total)
