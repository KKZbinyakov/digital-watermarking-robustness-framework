from src.dwarf.ready_solutions.utils.utils import *

class BER(Ready_Robustness_Expertise):
    @staticmethod
    def expertise(args: dict = {"original_bits": None, "extracted_bits": None}):
        """
        Вычисляет Bit Error Rate (BER) между двумя битовыми строками.
        Если длины не совпадают, сравнение происходит по минимальной длине.
        :param original_bits: строка из '0' и '1' (оригинальный ЦВЗ)
        :param extracted_bits: строка из '0' и '1' (извлечённый ЦВЗ)
        :return: значение BER (float от 0 до 1)
        """
        original_bits = args["original_bits"]
        extracted_bits = args["extracted_bits"]
        min_len = min(len(original_bits), len(extracted_bits))
        if min_len == 0:
            return 1.0  # или 0.0? По соглашению — 1.0 (полная ошибка)

        errors = 0
        o = int(original_bits[:min_len], 2)
        e = int(extracted_bits[:min_len], 2)
        errors = (o ^ e).bit_count()
        ber = errors / min_len
        return ber
    
    # и тд