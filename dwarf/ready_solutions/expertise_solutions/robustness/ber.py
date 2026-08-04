"""Метрика BER: доля неверно восстановленных бит ЦВЗ."""

import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Robustness_Expertise
from dwarf.ready_solutions.utils.expertise_utils import align_bits, bits_to_array


class BER(Ready_Robustness_Expertise):
    """
    Bit Error Rate — доля неверно восстановленных бит ЦВЗ.

    Основная метрика стойкости: 0 означает точное восстановление, 0.5 —
    результат, неотличимый от случайного угадывания, 1 — полную инверсию.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает долю несовпадающих бит между исходным и извлечённым ЦВЗ.

        Args:
            args (dict): параметры метрики
                original_bits (str): исходный ЦВЗ, строка из символов '0' и '1'
                extracted_bits (str): извлечённый ЦВЗ, строка из символов '0' и '1'
                allow_length_mismatch (bool): сравнивать по общей части при разной длине (по умолчанию False)

        Returns:
            float: значение BER в диапазоне от 0 до 1

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

        original = bits_to_array(original_bits)
        extracted = bits_to_array(extracted_bits)
        return float(np.count_nonzero(original != extracted) / length)
