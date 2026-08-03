from ...utils.attack_utils import *

class Brightness_Contrast(Ready_Color_Brightness_Attacks):
    """
    Атака изменения яркости и контраста.

    Последовательно применяет масштабирование относительно чёрного (яркость) и
    относительно средней яркости кадра (контраст).
    """

    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Меняет яркость и контраст изображения и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                brightness (float): множитель яркости (по умолчанию 1.2)
                contrast (float): множитель контраста (по умолчанию 1.2)

        Returns:
            None
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        brightness = float(args.get("brightness", 1.2))
        contrast = float(args.get("contrast", 1.2))

        img = Image.open(input_data).convert("RGB")
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img.save(output_data)