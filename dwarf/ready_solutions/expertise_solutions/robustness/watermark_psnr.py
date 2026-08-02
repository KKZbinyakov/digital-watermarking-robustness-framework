from ...utils.expertise_utils import *


class Watermark_PSNR(Ready_Robustness_Expertise):
    """
    PSNR между исходным и восстановленным изображением-знаком.

    Применяется к схемам, где ЦВЗ представляет собой картинку, а не битовую
    строку: там BER неприменим, а качество восстановления всё равно надо мерить.
    """

    @staticmethod
    def expertise(args: dict = {"original_watermark_path": None, "extracted_watermark_path": None}):
        """
        Считает PSNR между двумя изображениями-знаками по каналу яркости.

        Изображения обрезаются до общего размера, если он различается.

        Args:
            args (dict): параметры метрики
                original_watermark_path (str): путь к исходному изображению-знаку
                extracted_watermark_path (str): путь к восстановленному изображению-знаку

        Returns:
            float: PSNR в децибелах, inf при полном совпадении
        """
        original = load_gray(args["original_watermark_path"])
        extracted = load_gray(args["extracted_watermark_path"])

        rows = min(original.shape[0], extracted.shape[0])
        cols = min(original.shape[1], extracted.shape[1])
        original = original[:rows, :cols]
        extracted = extracted[:rows, :cols]

        mean_squared_error = np.mean((original - extracted) ** 2)
        if mean_squared_error == 0:
            return float("inf")
        return float(10 * np.log10(255**2 / mean_squared_error))
