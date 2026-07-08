import numpy as np
from PIL import Image

def extract_lsb(image_path, num_bits):
    """
    Извлекает указанное количество бит из младших бит всех каналов RGB.
    :param image_path: путь к изображению (с встроенным ЦВЗ)
    :param num_bits: сколько бит извлечь
    :return: строка из '0' и '1'
    """
    img = Image.open(image_path).convert('RGB')
    data = np.array(img)
    flat = data.ravel()
    extracted = flat[:num_bits] & 1
    bits_str = ''.join(str(b) for b in extracted)
    return bits_str