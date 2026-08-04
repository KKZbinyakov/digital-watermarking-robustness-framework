from PIL import Image

from dwarf.core.attack_orchestrator.attack_core import Ready_Compression_Attacks
from dwarf.ready_solutions.utils.attack_utils import roundtrip_buffer


class Webp(Ready_Compression_Attacks):
    """
    Атака сжатия WebP в режиме с потерями или без.
    """

    @staticmethod
    def attack(**args):
        """
        Пережимает изображение кодеком WebP и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                quality (int): качество для режима с потерями, 0..100 (по умолчанию 75)
                lossless (bool): сжатие без потерь, при нём quality игнорируется кодеком (по умолчанию False)

        Returns:
            None
        """
        defaults = {"input_data": None, "quality": 75, "lossless": False}
        args = {**defaults, **args}
        input_data = args["input_data"]
        output_data = args["output_data"]
        quality = int(args.get("quality", 75))
        lossless = bool(args.get("lossless", False))

        img = Image.open(input_data).convert("RGB")
        roundtrip_buffer(img, "WEBP", quality=quality, lossless=lossless).save(output_data)
