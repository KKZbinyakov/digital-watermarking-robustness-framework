from ..utils.expertise_utils import *


class Precision(Ready_Robustness_Expertise):
    """
    Доля верных срабатываний среди всех срабатываний детектора.

    Отвечает на вопрос, насколько можно доверять утверждению «знак есть».
    Для доказательства авторства важнее recall: ложное обвинение дороже пропуска.
    """

    @staticmethod
    def expertise(args: dict = {
        "y_true": None,
        "y_pred": None
    }):
        """
        Считает точность бинарного детектора.

        При полном отсутствии срабатываний возвращает 0.0: делить не на что,
        а нейтральная единица завысила бы оценку молчащего детектора.

        Args:
            args (dict): параметры метрики
                y_true: истинные метки, последовательность из 0 и 1
                y_pred: предсказанные метки, последовательность из 0 и 1

        Returns:
            float: значение precision в диапазоне от 0 до 1

        Raises:
            ValueError: если формы меток не совпадают или массивы пусты
        """
        true_positive, _, false_positive, _ = confusion_counts(args["y_true"], args["y_pred"])
        detected = true_positive + false_positive
        return float(true_positive / detected) if detected else 0.0
