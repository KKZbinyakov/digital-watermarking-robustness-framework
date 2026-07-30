from ...utils.expertise_utils import *


class NIQE(Ready_Imperceptibility_Expertise):
    """
    Natural Image Quality Evaluator — безэталонная оценка качества изображения.

    Оценивает естественность статистик изображения относительно модели, обученной на неискажённых снимках. Меньшее значение означает лучшее качество.

    Безэталонная: оригинал не нужен, поэтому применима там, где его нет, например к изображению после неизвестной обработки.

    Требует пакет pyiqa, не входящий в базовые зависимости dwarf. Проверка
    перенесена внутрь expertise, чтобы отсутствие пакета не ломало импорт
    модуля и не выбивало метрику из реестра.
    """

    @staticmethod
    def expertise(args: dict = {
        "image_path": None
    }):
        """
        Считает NIQE для одного изображения.

        Args:
            args (dict): параметры метрики
                image_path (str): путь к оцениваемому изображению

        Returns:
            float: значение NIQE, меньше — лучше

        Raises:
            RuntimeError: если пакет pyiqa не установлен
        """
        metric = iqa_metric("niqe")
        return float(metric(to_tensor(args["image_path"])).item())
