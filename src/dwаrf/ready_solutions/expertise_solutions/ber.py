def compute_ber(original_bits, extracted_bits):
    """
    Вычисляет Bit Error Rate (BER) между двумя битовыми строками.
    Если длины не совпадают, сравнение происходит по минимальной длине.
    :param original_bits: строка из '0' и '1' (оригинальный ЦВЗ)
    :param extracted_bits: строка из '0' и '1' (извлечённый ЦВЗ)
    :return: значение BER (float от 0 до 1)
    """
    min_len = min(len(original_bits), len(extracted_bits))
    if min_len == 0:
        return 1.0  # или 0.0? По соглашению — 1.0 (полная ошибка)

    errors = 0
    for i in range(min_len):
        if original_bits[i] != extracted_bits[i]:
            errors += 1
    ber = errors / min_len
    return ber