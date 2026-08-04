from PIL import Image

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
        Пережимает изображение кодеком HEIF и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                quality (int): качество, 0..100 (по умолчанию 50)

        Returns:
            None

        Raises:
            RuntimeError: если пакет pillow-heif не установлен
        """
        defaults = {"input_data": None, "quality": 50}
        args = {**defaults, **args}
        input_data = args["input_data"]
        output_data = args["output_data"]
        quality = int(args.get("quality", 50))

        try:
            import pillow_heif
        except ImportError as error:
            raise RuntimeError(
                "Для атаки Heic нужен пакет pillow-heif: pip install pillow-heif"
            ) from error
        pillow_heif.register_heif_opener()

        img = Image.open(input_data).convert("RGB")
        roundtrip_buffer(img, "HEIF", quality=quality).save(output_data)
