"""Метрика DISTS: нейросетевое сравнение структуры и текстуры раздельно."""

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Imperceptibility_Expertise
from dwarf.ready_solutions.utils.expertise_utils import iqa_metric, to_tensor


class DISTS(Ready_Imperceptibility_Expertise):
    """
    Deep Image Structure and Texture Similarity — нейросетевая метрика, сравнивающая структуру и текстуру раздельно.

    Меньшее значение означает меньшее различие. В отличие от LPIPS терпима к перестановке текстуры, поэтому не штрафует зря атаки вроде передискретизации.

    Требует пакет pyiqa, не входящий в базовые зависимости dwarf. Проверка
    перенесена внутрь expertise, чтобы отсутствие пакета не ломало импорт
    модуля и не выбивало метрику из реестра.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает DISTS между двумя изображениями.

        Args:
            args (dict): параметры метрики
                original_image (np.ndarray): матрица оригинального изображения
                distorted_image (np.ndarray): матрица изображения со встроенным ЦВЗ или после атаки

        Returns:
            float: значение DISTS, ноль при полном совпадении

        Raises:
            RuntimeError: если пакет pyiqa не установлен
        """
        defaults = {"original_image": None, "distorted_image": None}
        args = {**defaults, **args}
        metric = iqa_metric("dists")
        return float(metric(to_tensor(args["distorted_image"]), to_tensor(args["original_image"])).item())
