from PIL import Image

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
        Пережимает изображение кодеком AVIF и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                quality (int): качество, 0..100 (по умолчанию 50)

        Returns:
            None

        Raises:
            RuntimeError: если не установлен ни pillow-avif-plugin, ни pillow-heif
        """
        defaults = {"input_data": None, "quality": 50}
        args = {**defaults, **args}
        input_data = args["input_data"]
        output_data = args["output_data"]
        quality = int(args.get("quality", 50))

        try:
            import pillow_avif  # noqa: F401
        except ImportError:
            try:
                import pillow_heif  # noqa: F401
                pillow_heif.register_avif_opener()
            except ImportError as error:
                raise RuntimeError(
                    "Для атаки Avif нужен pillow-avif-plugin или pillow-heif: "
                    "pip install pillow-avif-plugin"
                ) from error

        img = Image.open(input_data).convert("RGB")
        roundtrip_buffer(img, "AVIF", quality=quality).save(output_data)
