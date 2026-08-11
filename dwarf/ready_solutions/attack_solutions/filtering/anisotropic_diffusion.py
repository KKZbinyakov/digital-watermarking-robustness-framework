"""Атака анизотропной диффузией (фильтр Перона-Малик): сглаживает изображение, сохраняя резкие границы."""

import numpy as np

from dwarf.core.attack_orchestrator.attack_core import Ready_Filtering_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_matrix


class Anisotropic_Diffusion(Ready_Filtering_Attacks):
    """
    Атака анизотропной диффузией (фильтр Перона-Малик).

    Итеративно сглаживает изображение нелинейной диффузией: внутри однородных
    областей диффузия идёт почти как изотропное размытие и стирает мелкие
    детали и шум, а на резких границах коэффициент проводимости падает почти
    до нуля и граница сохраняется. За счёт этого фильтр подавляет шум и
    мелкоструктурный водяной знак, искажая крупные контуры заметно меньше,
    чем гауссово или прямоугольное размытие той же силы.
    """

    @staticmethod
    def attack(**args):
        """
        Применяет анизотропную диффузию Перона-Малик к изображению.

        Диффузия ведётся одновременно по всем трём каналам RGB на кадре,
        нормированном к диапазону 0..1. Края кадра обрабатываются отражением
        (продолжением крайних отсчётов), без циклического оборачивания.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                iterations (int): число итераций диффузии, диапазон [1, 50] (по умолчанию 15)
                kappa (float): порог проводимости в долях канала 0..1, диапазон [0.01, 0.5] (по умолчанию 0.1)
                gamma (float): шаг диффузии за итерацию, диапазон (0.0, 0.25] (по умолчанию 0.2)
                option (int): функция проводимости: 1 — экспоненциальная, сильнее сохраняет резкие
                    границы; 2 — рациональная, слабее подавляет широкие плавные перепады (по умолчанию 1)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если iterations вне диапазона [1, 50], kappa вне диапазона [0.01, 0.5],
                gamma вне диапазона (0.0, 0.25] или option не равен 1 или 2
        """
        defaults = {"input_image": None, "iterations": 15, "kappa": 0.1, "gamma": 0.2, "option": 1}
        args = {**defaults, **args}
        input_image = args["input_image"]
        iterations = int(args["iterations"])
        kappa = float(args["kappa"])
        gamma = float(args["gamma"])
        option = int(args["option"])

        if not 1 <= iterations <= 50:
            raise ValueError(f"iterations must be in the range 1..50, got {iterations}")
        if not 0.01 <= kappa <= 0.5:
            raise ValueError(f"kappa must be in the range [0.01, 0.5], got {kappa}")
        if not 0.0 < gamma <= 0.25:
            raise ValueError(f"gamma must be in the range (0.0, 0.25], got {gamma}")
        if option not in (1, 2):
            raise ValueError(f"option must be 1 or 2, got {option}")

        data = to_matrix(input_image).astype(np.float32) / 255.0

        for _ in range(iterations):
            padded = np.pad(data, ((1, 1), (1, 1), (0, 0)), mode="edge")

            north = padded[:-2, 1:-1, :] - data
            south = padded[2:, 1:-1, :] - data
            east = padded[1:-1, 2:, :] - data
            west = padded[1:-1, :-2, :] - data

            if option == 1:
                c_n = np.exp(-((north / kappa) ** 2))
                c_s = np.exp(-((south / kappa) ** 2))
                c_e = np.exp(-((east / kappa) ** 2))
                c_w = np.exp(-((west / kappa) ** 2))
            else:
                c_n = 1.0 / (1.0 + (north / kappa) ** 2)
                c_s = 1.0 / (1.0 + (south / kappa) ** 2)
                c_e = 1.0 / (1.0 + (east / kappa) ** 2)
                c_w = 1.0 / (1.0 + (west / kappa) ** 2)

            data = data + gamma * (c_n * north + c_s * south + c_e * east + c_w * west)

        return to_matrix(np.clip(data, 0, 1) * 255.0)
