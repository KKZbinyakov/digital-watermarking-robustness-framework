from ..utils.expertise_utils import *


class PSNR(Ready_Imperceptibility_Expertise):
    """
    Пиковое отношение сигнала к шуму между двумя изображениями.
    """

    @staticmethod
    def expertise(args: dict = {
        "original_path": None,
        "distorted_path": None
    }):
        """
        Считает PSNR между двумя изображениями по каналам RGB.

        Args:
            args (dict): параметры метрики
                original_path (str): путь к оригинальному изображению
                distorted_path (str): путь к изображению со встроенным ЦВЗ или после атаки

        Returns:
            float: PSNR в децибелах, inf при полном совпадении
        """
        original = load_rgb_float(args["original_path"])
        distorted = load_rgb_float(args["distorted_path"])
        mean_squared_error = np.mean((original - distorted) ** 2)
        if mean_squared_error == 0:
            return float("inf")
        return float(10 * np.log10(255 ** 2 / mean_squared_error))
