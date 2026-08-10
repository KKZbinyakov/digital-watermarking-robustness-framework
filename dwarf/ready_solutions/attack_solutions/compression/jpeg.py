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

        Returns:
            np.ndarray: матрица изображения после атаки
        """
        defaults = {"input_image": None, "quality": 75}
        args = {**defaults, **args}
        input_image = args["input_image"]
        quality = int(args["quality"])

        return roundtrip_buffer(input_image, "JPEG", quality=quality)
