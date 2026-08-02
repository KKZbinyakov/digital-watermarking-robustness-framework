from ...utils.expertise_utils import *


class MSE(Ready_Imperceptibility_Expertise):
    """
    Среднеквадратичная ошибка по каналам RGB.
    """

    @staticmethod
    def expertise(args: dict = {"original_path": None, "distorted_path": None}):
        """
        Считает среднеквадратичную ошибку между двумя изображениями.

        Args:
            args (dict): параметры метрики
                original_path (str): путь к оригинальному изображению
                distorted_path (str): путь к изображению со встроенным ЦВЗ или после атаки

        Returns:
            float: значение MSE, не меньше нуля
        """
        original = load_rgb_float(args["original_path"])
        distorted = load_rgb_float(args["distorted_path"])
        return float(np.mean((original - distorted) ** 2))
