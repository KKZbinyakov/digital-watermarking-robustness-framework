"""Атака медианной фильтрацией: заменяет пиксели медианой соседей."""

from PIL import ImageFilter

from dwarf.core.attack_orchestrator.attack_core import Ready_Filtering_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_array, to_pil


class Median_Filter(Ready_Filtering_Attacks):
    """
    Атака медианной фильтрацией.

    Заменяет каждый пиксель медианой значений в квадратном окне соседних
    пикселей, эффективно подавляя импульсный шум и мелкие детали водяного знака.
    """

    @staticmethod
    def attack(**args):
        """
        Применяет медианный фильтр к изображению.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                window (int): размер окна фильтра, одно из значений 3, 5 или 7 (по умолчанию 3)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если window не равен 3, 5 или 7
        """
        defaults = {"input_image": None, "window": 3}
        args = {**defaults, **args}
        input_image = args["input_image"]
        window = args["window"]

        if window not in (3, 5, 7):
            raise ValueError(f"window must be 3, 5 or 7, got {window}")

        img = to_pil(input_image)
        filtered_img = img.filter(ImageFilter.MedianFilter(size=window))
        return to_array(filtered_img)
