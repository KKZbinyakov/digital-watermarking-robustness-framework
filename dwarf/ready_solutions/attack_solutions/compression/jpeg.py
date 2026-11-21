"""Атака сжатия JPEG: блочное ДКП с квантованием коэффициентов."""

from dwarf.core.attack_orchestrator.attack_core import Ready_Compression_Attacks
from dwarf.ready_solutions.utils.attack_utils import roundtrip_buffer


class Jpeg(Ready_Compression_Attacks):
    """
    Атака JPEG-сжатия.
    """

    @staticmethod
    def attack(**args):
        """
        Пережимает изображение кодеком JPEG.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                quality (int): коэффициент качества, обычно 10..95 (по умолчанию 75)
                grayscale (bool): кодировать как одноканальное изображение
                    (по умолчанию False)
                return_stats (bool): вернуть размер потока и бит/пиксель
                    вместе с изображением (по умолчанию False)

        Returns:
            np.ndarray | tuple: матрица после атаки; при return_stats
                также бит/пиксель и число байт потока
        """
        defaults = {"input_image": None, "quality": 75, "grayscale": False, "return_stats": False}
        args = {**defaults, **args}
        input_image = args["input_image"]
        quality = int(args["quality"])

        return roundtrip_buffer(input_image, "JPEG", quality=quality,
                                grayscale=bool(args["grayscale"]),
                                return_stats=bool(args["return_stats"]))
