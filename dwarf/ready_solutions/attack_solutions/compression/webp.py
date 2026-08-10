"""Атака сжатия WebP в режиме с потерями или без."""

from dwarf.core.attack_orchestrator.attack_core import Ready_Compression_Attacks
from dwarf.ready_solutions.utils.attack_utils import roundtrip_buffer


class Webp(Ready_Compression_Attacks):
    """
    Атака сжатия WebP в режиме с потерями или без.
    """

    @staticmethod
    def attack(**args):
        """
        Пережимает изображение кодеком WebP.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                quality (int): качество для режима с потерями, 0..100 (по умолчанию 75)
                lossless (bool): сжатие без потерь, при нём quality игнорируется кодеком (по умолчанию False)

        Returns:
            np.ndarray: матрица изображения после атаки
        """
        defaults = {"input_image": None, "quality": 75, "lossless": False}
        args = {**defaults, **args}
        input_image = args["input_image"]
        quality = int(args["quality"])
        lossless = bool(args["lossless"])

        return roundtrip_buffer(input_image, "WEBP", quality=quality, lossless=lossless)
