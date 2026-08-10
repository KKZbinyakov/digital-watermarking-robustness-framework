"""Атака обесцвечивания: сводит кадр к оттенкам серого."""

from dwarf.core.attack_orchestrator.attack_core import Ready_Color_Brightness_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_array, to_pil


class Grayscale(Ready_Color_Brightness_Attacks):
    """
    Атака обесцвечивания.

    Переводит изображение в оттенки серого и возвращает обратно в три канала.
    """

    @staticmethod
    def attack(**args):
        """
        Обесцвечивает изображение.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения

        Returns:
            np.ndarray: матрица изображения после атаки
        """
        defaults = {"input_image": None}
        args = {**defaults, **args}
        input_image = args["input_image"]

        return to_array(to_pil(input_image).convert("L"))
