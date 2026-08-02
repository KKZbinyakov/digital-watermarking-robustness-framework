from ...utils.expertise_utils import *


class BER(Ready_Robustness_Expertise):
    """
    Bit Error Rate — доля неверно восстановленных бит ЦВЗ.

    Основная метрика стойкости: 0 означает точное восстановление, 0.5 —
    результат, неотличимый от случайного угадывания, 1 — полную инверсию.
    """

    @staticmethod
    def expertise(
        args: dict = {
            "original_bits": "",
            "extracted_bits": "",
            "allow_length_mismatch": False,
        },
    ):
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
        original_bits, extracted_bits, length = align_bits(
            args["original_bits"],
            args["extracted_bits"],
            bool(args.get("allow_length_mismatch", False)),
        )
        if length == 0:
            raise ValueError("нечего сравнивать: обе битовые строки пусты")
        return float(np.count_nonzero(original_bits != extracted_bits) / length)
