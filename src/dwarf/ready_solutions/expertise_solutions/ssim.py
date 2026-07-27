from ..utils.expertise_utils import *


class SSIM(Ready_Imperceptibility_Expertise):
    """
    Индекс структурного сходства, усреднённый по кадру.
    """

    @staticmethod
    def expertise(args: dict = {
        "original_path": None,
        "distorted_path": None
    }):
        """
        Считает средний по кадру SSIM между двумя изображениями.

        Args:
            args (dict): параметры метрики
                original_path (str): путь к оригинальному изображению
                distorted_path (str): путь к изображению со встроенным ЦВЗ или после атаки

        Returns:
            float: значение SSIM в диапазоне от -1 до 1, единица при полном совпадении
        """
        original = load_gray(args["original_path"])
        distorted = load_gray(args["distorted_path"])
        return float(ssim_maps(original, distorted)[0].mean())
