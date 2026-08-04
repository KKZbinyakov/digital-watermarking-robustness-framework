from PIL import Image

from dwarf.core.attack_orchestrator.attack_core import Ready_Compression_Attacks
from dwarf.ready_solutions.utils.attack_utils import roundtrip_buffer


class Jpeg2000(Ready_Compression_Attacks):
    """
    Атака сжатия JPEG2000 вейвлет-преобразованием 9/7.
    """

    @staticmethod
    def attack(**args):
        """
        Пережимает изображение кодеком JPEG2000 и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                compression_ratio (float): степень сжатия n:1, больше значение — сильнее сжатие (по умолчанию 20)

        Returns:
            None
        """
        defaults = {"input_data": None, "compression_ratio": 20}
        args = {**defaults, **args}
        input_data = args["input_data"]
        output_data = args["output_data"]
        compression_ratio = float(args.get("compression_ratio", 20))

        img = Image.open(input_data).convert("RGB")
        roundtrip_buffer(
            img, "JPEG2000",
            quality_mode="rates",
            quality_layers=[compression_ratio],
            irreversible=True,
        ).save(output_data)
