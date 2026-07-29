from ..utils.expertise_utils import *


class DISTS(Ready_Imperceptibility_Expertise):
    """
    Deep Image Structure and Texture Similarity — нейросетевая метрика, сравнивающая структуру и текстуру раздельно.

    Меньшее значение означает меньшее различие. В отличие от LPIPS терпима к перестановке текстуры, поэтому не штрафует зря атаки вроде передискретизации.

    Требует пакет pyiqa, не входящий в базовые зависимости dwarf. Проверка
    перенесена внутрь expertise, чтобы отсутствие пакета не ломало импорт
    модуля и не выбивало метрику из реестра.
    """

    @staticmethod
    def expertise(args: dict = {
        "original_path": None,
        "distorted_path": None
    }):
        """
        Считает DISTS между двумя изображениями.

        Args:
            args (dict): параметры метрики
                original_path (str): путь к оригинальному изображению
                distorted_path (str): путь к изображению со встроенным ЦВЗ или после атаки

        Returns:
            float: значение DISTS, ноль при полном совпадении

        Raises:
            RuntimeError: если пакет pyiqa не установлен
        """
        metric = iqa_metric("dists")
        return float(metric(to_tensor(args["distorted_path"]),
                            to_tensor(args["original_path"])).item())
