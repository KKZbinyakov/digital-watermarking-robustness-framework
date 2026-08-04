from PIL import Image

from dwarf.core.attack_orchestrator.attack_core import Ready_Compression_Attacks
from dwarf.ready_solutions.utils.attack_utils import roundtrip_buffer


class Jpeg(Ready_Compression_Attacks):
    """
    Атака JPEG-сжатия.
    """

    @staticmethod
    def attack(**args):
        """
        Пережимает изображение кодеком JPEG и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                quality (int): коэффициент качества, обычно 10..95 (по умолчанию 75)

        Returns:
            None
        """
        defaults = {"input_data": None, "quality": 75}
        args = {**defaults, **args}
        input_data = args["input_data"]
        output_data = args["output_data"]
        quality = int(args.get("quality", 75))

        img = Image.open(input_data).convert("RGB")
        roundtrip_buffer(img, "JPEG", quality=quality).save(output_data)
