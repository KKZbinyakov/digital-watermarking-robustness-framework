from ..utils.attack_utils import *


class Grayscale(Ready_Color_Brightness_Attacks):
    """
    Атака обесцвечивания.

    Переводит изображение в оттенки серого и возвращает обратно в три канала.
    """

    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Обесцвечивает изображение и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата

        Returns:
            None
        """
        input_data = args["input_data"]
        output_data = args["output_data"]

        Image.open(input_data).convert("L").convert("RGB").save(output_data)
