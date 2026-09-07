"""Атака гомоморфной фильтрацией: разделяет и по-разному усиливает частотные составляющие яркости."""

import numpy as np
from PIL import Image

from dwarf.core.attack_orchestrator.attack_core import Ready_Filtering_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_array, to_pil


class Homomorphic_Filter(Ready_Filtering_Attacks):
    """
    Атака гомоморфной фильтрацией.

    Разделяет яркостный канал на низкочастотную (освещённость) и
    высокочастотную (отражение) составляющие через логарифм и Фурье-образ,
    затем по-разному их усиливает, что переупорядочивает частотный состав
    изображения и подавляет спрятанный в нём водяной знак.
    """

    @staticmethod
    def attack(**args):
        """
        Применяет гомоморфную фильтрацию к яркостному каналу изображения.

        Работает в пространстве YCbCr: фильтруется только канал Y, каналы
        цветности не изменяются.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                gamma_low (float): коэффициент усиления низких частот, диапазон [0.1, 1.0] (по умолчанию 0.5)
                gamma_high (float): коэффициент усиления высоких частот, диапазон [1.0, 3.0] (по умолчанию 2.0)
                cutoff (float): частота среза фильтра, диапазон [10.0, 100.0] (по умолчанию 32.0)
                c (float): коэффициент крутизны перехода фильтра, диапазон (0.0, 10.0] (по умолчанию 1.0)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если gamma_low вне диапазона [0.1, 1.0], gamma_high вне диапазона [1.0, 3.0],
                cutoff вне диапазона [10.0, 100.0] или c вне диапазона (0.0, 10.0]
        """
        defaults = {"input_image": None, "gamma_low": 0.5, "gamma_high": 2.0, "cutoff": 32.0, "c": 1.0}
        args = {**defaults, **args}
        input_image = args["input_image"]
        gamma_low = float(args["gamma_low"])
        gamma_high = float(args["gamma_high"])
        cutoff = float(args["cutoff"])
        c = float(args["c"])

        if not 0.1 <= gamma_low <= 1.0:
            raise ValueError(f"gamma_low must be in the range [0.1, 1.0], got {gamma_low}")
        if not 1.0 <= gamma_high <= 3.0:
            raise ValueError(f"gamma_high must be in the range [1.0, 3.0], got {gamma_high}")
        if not 10.0 <= cutoff <= 100.0:
            raise ValueError(f"cutoff must be in the range [10.0, 100.0], got {cutoff}")
        if not 0.0 < c <= 10.0:
            raise ValueError(f"c must be in the range (0.0, 10.0], got {c}")

        ycbcr = np.asarray(to_pil(input_image).convert("YCbCr"), dtype=np.float32)
        y_channel = ycbcr[..., 0] / 255.0
        height, width = y_channel.shape

        log_y = np.log1p(y_channel)

        spectrum = np.fft.fftshift(np.fft.fft2(log_y))

        u = np.arange(height) - height / 2
        v = np.arange(width) - width / 2
        grid_v, grid_u = np.meshgrid(v, u)
        distance_sq = grid_u**2 + grid_v**2
        gain = (gamma_high - gamma_low) * (1.0 - np.exp(-c * distance_sq / (cutoff**2))) + gamma_low

        filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * gain))
        filtered = np.real(filtered)

        result_y = np.expm1(filtered)

        source_mean = float(y_channel.mean())
        result_mean = float(result_y.mean())
        if result_mean > 1e-8:
            result_y = result_y * (source_mean / result_mean)

        ycbcr[..., 0] = np.clip(result_y * 255.0, 0, 255)
        return to_array(Image.fromarray(ycbcr.astype(np.uint8), mode="YCbCr"))
