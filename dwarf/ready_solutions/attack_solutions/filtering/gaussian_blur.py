"""Атака гауссовым размытием: свёртывает изображение с гауссовым ядром."""

from PIL import ImageFilter

from dwarf.core.attack_orchestrator.attack_core import Ready_Filtering_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_array, to_pil


class Gaussian_Blur(Ready_Filtering_Attacks):
    """
    Атака гауссовым размытием.

    Свёртывает изображение с гауссовым ядром, подавляя высокочастотные
    детали, в которых обычно скрыт водяной знак.
    """

    @staticmethod
    def attack(**args):
        """
        Применяет гауссово размытие к изображению.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                sigma (float): радиус (стандартное отклонение) размытия, диапазон [0.5, 5.0] (по умолчанию 1.0)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если sigma вне диапазона [0.5, 5.0]
        """
        defaults = {"input_image": None, "sigma": 1.0}
        args = {**defaults, **args}
        input_image = args["input_image"]
        sigma = float(args["sigma"])

        if not 0.5 <= sigma <= 5.0:
            raise ValueError(f"sigma must be in the range [0.5, 5.0], got {sigma}")

        img = to_pil(input_image)
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return to_array(blurred_img)
