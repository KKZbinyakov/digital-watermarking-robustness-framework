from ..utils.attack_utils import *


class Gamma_Correction(Ready_Color_Brightness_Attacks):
    """
    Атака гамма-коррекции.

    Применяет поканальное степенное преобразование: значения gamma меньше
    единицы осветляют средние тона, больше единицы - затемняют
    """

    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Применяет гамма-коррекцию к изображению и сохраняет результат.

        Результат зависит только от исходного уровня пикселя, поэтому
        преобразование считается один раз на таблицу из 256 значений.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                gamma (float): коэффициент гаммы, больше нуля (по умолчанию 1.5)

        Returns:
            None

        Raises:
            ValueError: если gamma не положительна
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        gamma = float(args.get("gamma", 1.5))

        if gamma <= 0:
            raise ValueError(f"gamma должна быть больше нуля, получено {gamma}")

        levels = np.arange(256, dtype=np.float64) / 255.0
        lookup = np.clip(255.0 * levels ** gamma, 0, 255).round().astype(np.uint8)
        save_rgb(lookup[load_rgb(input_data)], output_data)
