from ...utils.expertise_utils import *


class LPIPS(Ready_Imperceptibility_Expertise):
    """
    Learned Perceptual Image Patch Similarity — нейросетевая метрика различия на признаках предобученной свёрточной сети.

    Меньшее значение означает меньшее различие. Существенно лучше PSNR и SSIM согласуется с оценками наблюдателей, но требует torch и весов модели.

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
        Считает LPIPS между двумя изображениями.

        Args:
            args (dict): параметры метрики
                original_path (str): путь к оригинальному изображению
                distorted_path (str): путь к изображению со встроенным ЦВЗ или после атаки

        Returns:
            float: значение LPIPS, ноль при полном совпадении

        Raises:
            RuntimeError: если пакет pyiqa не установлен
        """
        metric = iqa_metric("lpips")
        return float(metric(to_tensor(args["distorted_path"]),
                            to_tensor(args["original_path"])).item())
