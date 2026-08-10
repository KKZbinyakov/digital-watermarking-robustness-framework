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

        Returns:
            np.ndarray: матрица изображения после атаки
        """
        defaults = {"input_image": None, "compression_ratio": 20}
        args = {**defaults, **args}
        input_image = args["input_image"]
        compression_ratio = float(args["compression_ratio"])

        return roundtrip_buffer(
            input_image,
            "JPEG2000",
            quality_mode="rates",
            quality_layers=[compression_ratio],
            irreversible=True,
        )
