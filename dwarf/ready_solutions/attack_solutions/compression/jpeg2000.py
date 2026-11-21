"""Атака сжатия JPEG2000: вейвлет-преобразование 9/7 с заданной степенью сжатия."""

from dwarf.core.attack_orchestrator.attack_core import Ready_Compression_Attacks
from dwarf.ready_solutions.utils.attack_utils import roundtrip_buffer


class Jpeg2000(Ready_Compression_Attacks):
    """
    Атака сжатия JPEG2000 вейвлет-преобразованием 9/7.
    """

    @staticmethod
    def attack(**args):
        """
        Пережимает изображение кодеком JPEG2000.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                compression_ratio (float): степень сжатия n:1, больше значение — сильнее сжатие (по умолчанию 20)
                grayscale (bool): кодировать как одноканальное изображение
                    (по умолчанию False)
                irreversible (bool): использовать необратимое преобразование
                    (по умолчанию True)
                return_stats (bool): вернуть размер потока и бит/пиксель
                    вместе с изображением (по умолчанию False)

        Returns:
            np.ndarray | tuple: матрица после атаки; при return_stats
                также бит/пиксель и число байт потока
        """
        defaults = {
            "input_image": None,
            "compression_ratio": 20,
            "grayscale": False,
            "return_stats": False,
            "irreversible": True,
        }
        args = {**defaults, **args}
        input_image = args["input_image"]
        compression_ratio = float(args["compression_ratio"])

        return roundtrip_buffer(
            input_image,
            "JPEG2000",
            quality_mode="rates",
            quality_layers=[compression_ratio],
            irreversible=bool(args["irreversible"]),
            grayscale=bool(args["grayscale"]),
            return_stats=bool(args["return_stats"]),
        )
