import numpy as np

from dwarf.core.expertise_orchestrator.expertise_core import Ready_Robustness_Expertise
from dwarf.ready_solutions.utils.expertise_utils import align_bits, bits_to_pm1


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
                original_bits (str): исходный ЦВЗ, строка из символов '0' и '1'
                extracted_bits (str): извлечённый ЦВЗ, строка из символов '0' и '1'
                allow_length_mismatch (bool): сравнивать по общей части при разной длине (по умолчанию False)

        Returns:
            float: значение NC в диапазоне от -1 до 1

        Raises:
            ValueError: если длины различаются без allow_length_mismatch, либо обе строки пусты
        """
        defaults = {"original_bits": None, "extracted_bits": None, "allow_length_mismatch": False}
        args = {**defaults, **args}
        original_bits, extracted_bits, length = align_bits(
            args["original_bits"], args["extracted_bits"],
            bool(args.get("allow_length_mismatch", False)),
        )
        if length == 0:
            raise ValueError("нечего сравнивать: обе битовые строки пусты")

        original = bits_to_pm1(original_bits)
        extracted = bits_to_pm1(extracted_bits)
        denominator = np.sqrt((original * original).sum() * (extracted * extracted).sum())
        return float((original * extracted).sum() / denominator) if denominator else 0.0
