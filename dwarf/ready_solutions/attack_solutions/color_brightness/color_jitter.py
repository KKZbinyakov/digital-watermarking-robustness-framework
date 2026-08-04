"""Атака случайного цветового дрожания."""

import numpy as np
from PIL import Image, ImageEnhance

from dwarf.core.attack_orchestrator.attack_core import Ready_Color_Brightness_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_array, to_pil


class Color_Jitter(Ready_Color_Brightness_Attacks):
    """
    Атака случайного цветового дрожания.

    Случайно меняет яркость, контраст, насыщенность и оттенок в пределах
    заданного размаха, моделируя нестабильную цветокоррекцию при публикации
    через разные сервисы. Каждый параметр задаёт полуширину равномерного
    отклонения от исходного значения.
    """

    @staticmethod
    def attack(**args):
        """
        Применяет случайное цветовое дрожание.

        Оттенок сдвигается в пространстве HSV, где канал H занимает диапазон
        0..255 и замкнут по кругу, поэтому сдвиг берётся по модулю 256.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                brightness (float): размах яркости, множитель 1 +- brightness (по умолчанию 0.2)
                contrast (float): размах контраста, множитель 1 +- contrast (по умолчанию 0.2)
                saturation (float): размах насыщенности, множитель 1 +- saturation (по умолчанию 0.2)
                hue (float): размах сдвига оттенка в долях цветового круга (по умолчанию 0.05)
                seed (int): зерно генератора случайных чисел, без него атака невоспроизводима (по умолчанию None)

        Returns:
            np.ndarray: матрица изображения после атаки
        """
        defaults = {
            "input_image": None,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.05,
            "seed": None,
        }
        args = {**defaults, **args}
        input_image = args["input_image"]
        brightness = float(args["brightness"])
        contrast = float(args["contrast"])
        saturation = float(args["saturation"])
        hue = float(args["hue"])
        seed = args["seed"]

        rng = np.random.default_rng(seed)
        img = to_pil(input_image)
        img = ImageEnhance.Brightness(img).enhance(1 + rng.uniform(-brightness, brightness))
        img = ImageEnhance.Contrast(img).enhance(1 + rng.uniform(-contrast, contrast))
        img = ImageEnhance.Color(img).enhance(1 + rng.uniform(-saturation, saturation))

        hsv = np.asarray(img.convert("HSV")).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.uniform(-hue, hue) * 255)) % 256
        return to_array(Image.fromarray(hsv.astype(np.uint8), "HSV"))
