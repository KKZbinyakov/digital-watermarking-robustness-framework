from dwarf.ready_solutions.utils.attack_utils import *
from PIL import ImageFilter

class Median_Filter(Ready_Filtering_Attacks):
    """
    Атака медианной фильтрацией.

    Заменяет каждый пиксель медианой значений в квадратном окне соседних
    пикселей, эффективно подавляя импульсный шум и мелкие детали водяного знака.
    """

    @staticmethod
    def attack(args: dict = {
                "input_image": [[[0]]],
                "window": 3
    }):
        """
        Применяет медианный фильтр к изображению и возвращает результат.

        Args:
            args (dict): параметры атаки
                input_image (list(list(list(int)))): матрица изображения
                window (int): размер окна фильтра, одно из значений 3, 5 или 7 (по умолчанию 3)

        Returns:
            output_image (list(list(list(int)))): матрица изображения после атаки

        Raises:
            ValueError: если window не равен 3, 5 или 7
        """
        input_image = args["input_image"]
        window = args.get("window", 3)

        if window not in (3, 5, 7):
            raise ValueError("Window size must be 3, 5 or 7")

        img = Image.fromarray(np.array(input_image, dtype=np.uint8), "RGB")
        filtered_img = img.filter(ImageFilter.MedianFilter(size=window))
        return np.array(filtered_img).tolist()