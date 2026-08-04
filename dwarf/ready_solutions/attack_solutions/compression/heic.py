"""Атака сжатия HEIC: кодек HEVC через pillow-heif."""

from dwarf.core.attack_orchestrator.attack_core import Ready_Compression_Attacks
from dwarf.ready_solutions.utils.attack_utils import roundtrip_buffer


class Heic(Ready_Compression_Attacks):
    """
    Атака сжатия HEIC / HEIF кодеком HEVC.

    Формат по умолчанию используется камерами Apple, поэтому атака моделирует
    передачу изображения через экосистему iOS.

    Требует пакет pillow-heif
    """

    @staticmethod
    def attack(**args):
        """
        Пережимает изображение кодеком HEIF.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                quality (int): качество, 0..100 (по умолчанию 50)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            RuntimeError: если пакет pillow-heif не установлен
        """
        defaults = {"input_image": None, "quality": 50}
        args = {**defaults, **args}
        input_image = args["input_image"]
        quality = int(args["quality"])

        try:
            import pillow_heif
        except ImportError as error:
            raise RuntimeError("Heic attack requires the pillow-heif package: pip install pillow-heif") from error
        pillow_heif.register_heif_opener()

        return roundtrip_buffer(input_image, "HEIF", quality=quality)
