from dwarf.core.expertise_orchestrator.expertise_core import Ready_Imperceptibility_Expertise
from dwarf.ready_solutions.utils.expertise_utils import load_gray, ssim_maps


class SSIM(Ready_Imperceptibility_Expertise):
    """
    Индекс структурного сходства, усреднённый по кадру.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает средний по кадру SSIM между двумя изображениями.

        Args:
            args (dict): параметры метрики
                original_path (str): путь к оригинальному изображению
                distorted_path (str): путь к изображению со встроенным ЦВЗ или после атаки

        Returns:
            float: значение SSIM в диапазоне от -1 до 1, единица при полном совпадении
        """
        defaults = {} # Написать дефолтные значения
        args = {**defaults, **args}
        original = load_gray(args["original_path"])
        distorted = load_gray(args["distorted_path"])
        return float(ssim_maps(original, distorted)[0].mean())
