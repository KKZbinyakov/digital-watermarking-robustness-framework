"""Атака нерезким маскированием (unsharp mask): усиливает контраст на границах."""

from PIL import ImageFilter

from dwarf.core.attack_orchestrator.attack_core import Ready_Filtering_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_array, to_pil


class Unsharp_Mask(Ready_Filtering_Attacks):
    """
    Атака нерезким маскированием (unsharp mask).

    Усиливает контраст на границах, вычитая размытую версию изображения из
    исходной и добавляя разницу обратно с заданной силой, что искажает
    локальную структуру и может подавлять водяной знак.
    """

    @staticmethod
    def attack(**args):
        """
        Применяет нерезкое маскирование к изображению.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                amount (float): сила эффекта резкости, диапазон [0.1, 5.0] (по умолчанию 1.0)
                radius (float): радиус размытия маски, диапазон [0.5, 5.0] (по умолчанию 2.0)
                threshold (float): порог срабатывания в долях канала 0..1, диапазон [0.0, 1.0] (по умолчанию 0.0)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если amount вне диапазона [0.1, 5.0], radius вне диапазона [0.5, 5.0]
                или threshold вне диапазона [0.0, 1.0]
        """
        defaults = {"input_image": None, "amount": 1.0, "radius": 2.0, "threshold": 0.0}
        args = {**defaults, **args}
        input_image = args["input_image"]
        amount = float(args["amount"])
        radius = float(args["radius"])
        threshold = float(args["threshold"])

        if not 0.1 <= amount <= 5.0:
            raise ValueError(f"amount must be in the range [0.1, 5.0], got {amount}")
        if not 0.5 <= radius <= 5.0:
            raise ValueError(f"radius must be in the range [0.5, 5.0], got {radius}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in the range [0.0, 1.0], got {threshold}")

        img = to_pil(input_image)
        sharp_img = img.filter(
            ImageFilter.UnsharpMask(radius=radius, percent=int(amount * 100), threshold=int(threshold * 255))
        )
        return to_array(sharp_img)
