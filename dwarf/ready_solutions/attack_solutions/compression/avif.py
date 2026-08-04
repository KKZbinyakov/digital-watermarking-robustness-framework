"""Атака сжатия AVIF: кодек AV1 через pillow-avif-plugin или pillow-heif."""

from dwarf.core.attack_orchestrator.attack_core import Ready_Compression_Attacks
from dwarf.ready_solutions.utils.attack_utils import roundtrip_buffer


class Avif(Ready_Compression_Attacks):
    """
    Атака сжатия AVIF кодеком AV1.

    При том же визуальном качестве даёт заметно более сильное искажение, чем
    JPEG, поэтому полезна как верхняя граница по агрессивности сжатия.

    Требует pillow-avif-plugin либо pillow-heif
    """

    @staticmethod
    def attack(**args):
        """
        Пережимает изображение кодеком AVIF.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                quality (int): качество, 0..100 (по умолчанию 50)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            RuntimeError: если не установлен ни pillow-avif-plugin, ни pillow-heif
        """
        defaults = {"input_image": None, "quality": 50}
        args = {**defaults, **args}
        input_image = args["input_image"]
        quality = int(args["quality"])

        try:
            import pillow_avif  # noqa: F401
        except ImportError:
            try:
                import pillow_heif
            except ImportError as error:
                raise RuntimeError(
                    "Avif attack requires pillow-avif-plugin or pillow-heif: pip install pillow-avif-plugin"
                ) from error
            pillow_heif.register_avif_opener()

        return roundtrip_buffer(input_image, "AVIF", quality=quality)
