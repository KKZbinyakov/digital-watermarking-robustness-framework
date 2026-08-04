from PIL import Image, ImageEnhance

from dwarf.core.attack_orchestrator.attack_core import Ready_Color_Brightness_Attacks


class Brightness_Contrast(Ready_Color_Brightness_Attacks):
    """
    Атака изменения яркости и контраста.

    Последовательно применяет масштабирование относительно чёрного (яркость) и
    относительно средней яркости кадра (контраст).
    """

    @staticmethod
    def attack(**args):
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
        defaults = {"input_data": None, "brightness": 1.2, "contrast": 1.2}
        args = {**defaults, **args}
        input_data = args["input_data"]
        output_data = args["output_data"]
        brightness = float(args.get("brightness", 1.2))
        contrast = float(args.get("contrast", 1.2))

        img = Image.open(input_data).convert("RGB")
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img.save(output_data)
