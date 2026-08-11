"""Атака прямоугольным (box) размытием: усредняет пиксели по квадратному окну."""

from PIL import ImageFilter

from dwarf.core.attack_orchestrator.attack_core import Ready_Filtering_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_array, to_pil


class Box_Filter(Ready_Filtering_Attacks):
    """
    Атака прямоугольным (box) размытием.

    Заменяет каждый пиксель средним значением по квадратному окну соседних
    пикселей, равномерно размывая изображение и высокочастотный водяной знак.
    """

    @staticmethod
    def attack(**args):
        """
        Применяет прямоугольное размытие к изображению.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                window (int): размер окна усреднения, нечётное число >= 3 (по умолчанию 3)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если window меньше 3 или чётное
        """
        defaults = {"input_image": None, "window": 3}
        args = {**defaults, **args}
        input_image = args["input_image"]
        window = args["window"]

        if window < 3 or window % 2 == 0:
            raise ValueError(f"window must be an odd number >= 3, got {window}")

        img = to_pil(input_image)
        filtered_img = img.filter(ImageFilter.BoxBlur(radius=window // 2))
        return to_array(filtered_img)
