"""Метрика Bit Agreement: доля совпавших бит исходного и извлечённого ЦВЗ."""

import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Robustness_Expertise
from dwarf.ready_solutions.utils.expertise_utils import align_bits, bits_to_array


class Bit_Agreement(Ready_Robustness_Expertise):
    """
    Доля совпавших бит бинарного водяного знака.

    Высокие значения означают высокую стойкость, как в таблице 2 статьи DCT.
    Печатная формула NC с XOR в той статье задаёт противоположную величину BER.
    """

    @staticmethod
    def expertise(**args):
        """
        Считает долю совпавших бит исходного и извлечённого знака.

        Args:
            args (dict): параметры метрики
                original_bits (str | array-like): исходный бинарный ЦВЗ
                extracted_bits (str | array-like): извлечённый бинарный ЦВЗ
                allow_length_mismatch (bool): сравнивать общую часть при разной длине
                    (по умолчанию False)

        Returns:
            float: доля совпавших бит в диапазоне от 0 до 1

        Raises:
            ValueError: если длины различаются без разрешения либо общая часть пуста
        """
        defaults = {"original_bits": None, "extracted_bits": None, "allow_length_mismatch": False}
        args = {**defaults, **args}
        original, extracted, length = align_bits(
            args["original_bits"], args["extracted_bits"], bool(args["allow_length_mismatch"])
        )
        if length == 0:
            raise ValueError("nothing to compare: both bit strings are empty")
        matches = np.count_nonzero(bits_to_array(original) == bits_to_array(extracted))
        return float(matches / length)
