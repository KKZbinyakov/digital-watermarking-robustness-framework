from ..utils.attack_utils import *


class Jpeg(Ready_Compression_Attacks):
    """
    Атака JPEG-сжатия.
    """

    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
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
        input_data = args["input_data"]
        output_data = args["output_data"]
        quality = int(args.get("quality", 75))

        img = Image.open(input_data).convert("RGB")
        roundtrip_buffer(img, "JPEG", quality=quality).save(output_data)
