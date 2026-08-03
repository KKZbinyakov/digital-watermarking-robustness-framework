from ...utils.attack_utils import *


class Tiff(Ready_Compression_Attacks):
    """
    Атака перекодирования в TIFF с LZW или Deflate.

    Сжатие без потерь, пиксели не меняются
    """

    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Перекодирует изображение в TIFF и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                compression (str): 'tiff_lzw' или 'tiff_adobe_deflate' (по умолчанию 'tiff_lzw')

        Returns:
            None
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        compression = args.get("compression", "tiff_lzw")

        img = Image.open(input_data).convert("RGB")
        roundtrip_buffer(img, "TIFF", compression=compression).save(output_data)
