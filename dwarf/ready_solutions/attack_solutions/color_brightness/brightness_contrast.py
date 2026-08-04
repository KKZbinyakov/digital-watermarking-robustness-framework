"""Атака изменения яркости и контраста."""

from PIL import ImageEnhance

from dwarf.core.attack_orchestrator.attack_core import Ready_Color_Brightness_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_array, to_pil


class Brightness_Contrast(Ready_Color_Brightness_Attacks):
    """
    Атака изменения яркости и контраста.

    Последовательно применяет масштабирование относительно чёрного (яркость) и
    относительно средней яркости кадра (контраст).
    """

    @staticmethod
    def attack(**args):
        """
        Меняет яркость и контраст изображения.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                brightness (float): множитель яркости (по умолчанию 1.2)
                contrast (float): множитель контраста (по умолчанию 1.2)

        Returns:
            np.ndarray: матрица изображения после атаки
        """
        defaults = {"input_image": None, "brightness": 1.2, "contrast": 1.2}
        args = {**defaults, **args}
        input_image = args["input_image"]
        brightness = float(args["brightness"])
        contrast = float(args["contrast"])

        img = to_pil(input_image)
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        return to_array(img)
