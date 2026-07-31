from ...utils.attack_utils import *

class Median_Filter(Ready_Filtering_Attacks):
    """
    Атака медианной фильтрацией.

    Заменяет каждый пиксель медианой значений в квадратном окне соседних
    пикселей, эффективно подавляя импульсный шум и мелкие детали водяного знака.
    """

    @staticmethod
    def attack(args: dict = {
                "input_data": None,
                "output_data": None
    }):
        """
        Применяет медианный фильтр к изображению и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                window (int): размер окна фильтра, одно из значений 3, 5 или 7 (по умолчанию 3)

        Returns:
            None

        Raises:
            ValueError: если window не равен 3, 5 или 7
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        window = args.get("window", 3)
        
        if window not in (3, 5, 7):
            raise ValueError("Размер окна должен быть 3, 5 или 7")
            
        img = Image.open(input_data).convert("RGB")
        filtered_img = img.filter(ImageFilter.MedianFilter(size=window))
        filtered_img.save(output_data)