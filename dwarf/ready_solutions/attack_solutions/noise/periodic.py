"""Атака периодическим (синусоидальным) шумом: накладывает синусоидальную волну на изображение."""

import numpy as np

from dwarf.core.attack_orchestrator.attack_core import Ready_Noise_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_matrix


class Periodic(Ready_Noise_Attacks):
    """
    Атака периодическим (синусоидальным) шумом.

    Накладывает на изображение синусоидальную волну вдоль горизонтальной оси
    со случайной фазой, моделируя помехи развёртки или растрирования.
    """

    @staticmethod
    def attack(**args):
        """
        Накладывает синусоидальную помеху на изображение.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                amplitude (float): амплитуда волны в долях канала 0..1, диапазон [0.01, 0.2] (по умолчанию 0.05)
                frequency (float): частота волны по ширине изображения, диапазон [1.0, 100.0] (по умолчанию 10.0)
                seed (int): зерно генератора случайных чисел, определяет фазу волны (по умолчанию None)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если amplitude вне диапазона [0.01, 0.2] или frequency вне диапазона [1.0, 100.0]
        """
        defaults = {"input_image": None, "amplitude": 0.05, "frequency": 10.0, "seed": None}
        args = {**defaults, **args}
        input_image = args["input_image"]
        amplitude = float(args["amplitude"])
        frequency = float(args["frequency"])
        seed = args["seed"]

        if not 0.01 <= amplitude <= 0.2:
            raise ValueError(f"amplitude must be in the range [0.01, 0.2], got {amplitude}")
        if not 1.0 <= frequency <= 100.0:
            raise ValueError(f"frequency must be in the range [1.0, 100.0], got {frequency}")

        rng = np.random.default_rng(seed)
        data = to_matrix(input_image).astype(np.float32) / 255.0

        height, width, _ = data.shape
        phase = rng.uniform(0.0, 2 * np.pi)

        x = np.arange(width, dtype=np.float32)
        wave = amplitude * np.sin(2 * np.pi * frequency * x / width + phase)

        noisy = np.clip(data + wave[np.newaxis, :, np.newaxis], 0, 1) * 255.0
        return to_matrix(noisy)
