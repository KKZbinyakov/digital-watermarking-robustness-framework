"""Атака гауссовым размытием: свёртывает изображение с гауссовым ядром."""

import numpy as np
from PIL import ImageFilter
from scipy.ndimage import convolve1d

from dwarf.core.attack_orchestrator.attack_core import Ready_Filtering_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_array, to_matrix, to_pil


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
                window (int | None): нечётный размер ядра не меньше 3;
                    None использует стандартный фильтр Pillow (по умолчанию None)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если sigma вне диапазона [0.5, 5.0] либо window
                не является нечётным целым числом не меньше 3
        """
        defaults = {"input_image": None, "sigma": 1.0, "window": None}
        args = {**defaults, **args}
        input_image = args["input_image"]
        sigma = float(args["sigma"])
        window = args["window"]

        if not 0.5 <= sigma <= 5.0:
            raise ValueError(f"sigma must be in the range [0.5, 5.0], got {sigma}")
        if window is not None:
            if not isinstance(window, int) or window < 3 or window % 2 == 0:
                raise ValueError(f"window must be an odd integer >= 3, got {window}")
            coordinates = np.arange(-(window // 2), window // 2 + 1, dtype=np.float64)
            kernel = np.exp(-0.5 * (coordinates / sigma) ** 2)
            kernel /= kernel.sum()
            data = to_matrix(input_image).astype(np.float64)
            filtered = convolve1d(data, kernel, axis=0, mode="reflect")
            filtered = convolve1d(filtered, kernel, axis=1, mode="reflect")
            return to_matrix(filtered)

        img = to_pil(input_image)
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return to_array(blurred_img)
