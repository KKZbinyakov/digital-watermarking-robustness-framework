"""Атака уменьшения битовой глубины: отбрасывает младшие биты каналов."""

import numpy as np

from dwarf.core.attack_orchestrator.attack_core import Ready_Color_Brightness_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_matrix


class Bit_Depth_Reduction(Ready_Color_Brightness_Attacks):
    """
    Атака уменьшения битовой глубины.

    Оставляет только старшие биты каждого канала и растягивает полученные уровни
    обратно на диапазон 0..255
    """

    @staticmethod
    def attack(**args):
        """
        Огрубляет битовую глубину изображения.

        Результат зависит только от исходного уровня пикселя, поэтому
        преобразование считается один раз на таблицу из 256 значений.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                bits (int): сколько старших бит оставить, 1..8 (по умолчанию 4)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если bits вне диапазона 1..8
        """
        defaults = {"input_image": None, "bits": 4}
        args = {**defaults, **args}
        input_image = args["input_image"]
        bits = int(args["bits"])

        if not 1 <= bits <= 8:
            raise ValueError(f"bits must be in the range 1..8, got {bits}")

        shift = 8 - bits
        levels = (1 << bits) - 1
        lookup = (np.arange(256, dtype=np.uint16) >> shift) / levels * 255.0
        lookup = np.clip(lookup, 0, 255).round().astype(np.uint8)
        return lookup[to_matrix(input_image)]
