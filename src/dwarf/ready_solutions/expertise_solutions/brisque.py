from ..utils.expertise_utils import *


class BRISQUE(Ready_Imperceptibility_Expertise):
    """
    Blind/Referenceless Image Spatial Quality Evaluator — безэталонная оценка качества изображения.

    Оценивает качество по статистикам нормированной яркости в пространственной области. Меньшее значение означает лучшее качество.

    Безэталонная и заметно быстрее NIQE, но сильнее привязана к обучающей выборке искажений.

    Требует пакет pyiqa, не входящий в базовые зависимости dwarf. Проверка
    перенесена внутрь expertise, чтобы отсутствие пакета не ломало импорт
    модуля и не выбивало метрику из реестра.
    """

    @staticmethod
    def expertise(args: dict = {
        "image_path": None
    }):
        """
        Считает BRISQUE для одного изображения.

        Args:
            args (dict): параметры метрики
                image_path (str): путь к оцениваемому изображению

        Returns:
            float: значение BRISQUE, меньше — лучше

        Raises:
            RuntimeError: если пакет pyiqa не установлен
        """
        metric = iqa_metric("brisque")
        return float(metric(to_tensor(args["image_path"])).item())
