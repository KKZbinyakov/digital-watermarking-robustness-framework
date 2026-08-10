"""Метрика SSIM: индекс структурного сходства, усреднённый по кадру."""

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Imperceptibility_Expertise
from dwarf.ready_solutions.utils.expertise_utils import ssim_maps, to_gray


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
                original_image (np.ndarray): матрица оригинального изображения
                distorted_image (np.ndarray): матрица изображения со встроенным ЦВЗ или после атаки

        Returns:
            float: значение SSIM в диапазоне от -1 до 1, единица при полном совпадении

        Raises:
            ValueError: если кадр меньше окна свёртки
        """
        defaults = {"original_image": None, "distorted_image": None}
        args = {**defaults, **args}
        original = to_gray(args["original_image"])
        distorted = to_gray(args["distorted_image"])

        return float(ssim_maps(original, distorted)[0].mean())
