"""Метрика LPIPS: нейросетевая оценка различия на признаках свёрточной сети."""

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Imperceptibility_Expertise
from dwarf.ready_solutions.utils.expertise_utils import iqa_metric, to_tensor


class LPIPS(Ready_Imperceptibility_Expertise):
    """
    Learned Perceptual Image Patch Similarity — нейросетевая метрика различия на признаках предобученной свёрточной сети.

    Меньшее значение означает меньшее различие. Существенно лучше PSNR и SSIM согласуется с оценками наблюдателей, но требует torch и весов модели.

    Требует пакет pyiqa, не входящий в базовые зависимости dwarf. Проверка
    перенесена внутрь expertise, чтобы отсутствие пакета не ломало импорт
    модуля и не выбивало метрику из реестра.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает LPIPS между двумя изображениями.

        Args:
            args (dict): параметры метрики
                original_image (np.ndarray): матрица оригинального изображения
                distorted_image (np.ndarray): матрица изображения со встроенным ЦВЗ или после атаки

        Returns:
            float: значение LPIPS, ноль при полном совпадении

        Raises:
            RuntimeError: если пакет pyiqa не установлен
        """
        defaults = {"original_image": None, "distorted_image": None}
        args = {**defaults, **args}
        metric = iqa_metric("lpips")
        return float(metric(to_tensor(args["distorted_image"]), to_tensor(args["original_image"])).item())
