"""Метрика NC: нормированная корреляция исходного и извлечённого ЦВЗ."""

import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Robustness_Expertise
from dwarf.ready_solutions.utils.expertise_utils import align_bits, bits_to_array


class NC(Ready_Robustness_Expertise):
    """
    Normalized Correlation — нормированная корреляция исходного и извлечённого ЦВЗ.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает нормированную корреляцию между исходным и извлечённым ЦВЗ.

        Args:
            args (dict): параметры метрики
                original_bits (str | array-like): исходный бинарный ЦВЗ
                extracted_bits (str | array-like): извлечённый бинарный ЦВЗ
                allow_length_mismatch (bool): сравнивать по общей части при разной длине (по умолчанию False)

        Returns:
            float: значение NC в диапазоне от 0 до 1 для бинарных последовательностей 0/1

        Raises:
            ValueError: если длины различаются без allow_length_mismatch, либо обе строки пусты
        """
        defaults = {"original_bits": None, "extracted_bits": None, "allow_length_mismatch": False}
        args = {**defaults, **args}
        original_bits, extracted_bits, length = align_bits(
            args["original_bits"],
            args["extracted_bits"],
            bool(args["allow_length_mismatch"]),
        )
        if length == 0:
            raise ValueError("nothing to compare: both bit strings are empty")

        # Watermarking papers, including Wu et al. (IEEE Access, 2023),
        # define NC directly over binary pixels.  Mapping 0/1 to -1/+1
        # changes the reported values into 1 - 2 * BER and therefore makes
        # the framework results incomparable with the paper tables.
        original = bits_to_array(original_bits).astype(np.float64, copy=False)
        extracted = bits_to_array(extracted_bits).astype(np.float64, copy=False)
        denominator = np.sqrt((original * original).sum() * (extracted * extracted).sum())
        return float((original * extracted).sum() / denominator) if denominator else 0.0
