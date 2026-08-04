import numpy as np
from PIL import Image

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Spatial_Embeddings


class LSB(Ready_Spatial_Embeddings):
    @staticmethod
    def embedding(**args):
        """
        Встраивает биты ЦВЗ в младшие биты всех каналов RGB изображения.
        :param input_data: матрица входного изображения
        :param watermark_bits: строка из '0' и '1' (например, "101100...")

        :return output_image: матрица изображения с встроенным ЦВЗ
        """
        defaults = {"input_image": None, "watermark_bits": None}
        args = {**defaults, **args}
        input_image = args["image_path"]
        watermark_bits = args["watermark_bits"]
        img = Image.open(input_image).convert("RGB")
        data = np.array(img)  # форма (H, W, 3)
        flat = data.ravel()  # одномерный массив всех каналов

        bits = np.array(list(watermark_bits), dtype=np.uint8)
        max_len = len(flat)
        if len(bits) > max_len:
            bits = bits[:max_len]
            print(f"Предупреждение: ЦВЗ слишком длинный, обрезан до {max_len} бит")

        flat[: len(bits)] = (flat[: len(bits)] & 0xFE) | bits
        output_image = flat.reshape(data.shape)
        return output_image

    @staticmethod
    def extraction(**args):
        """
        Извлекает указанное количество бит из младших бит всех каналов RGB.
        :param image_path: путь к изображению (с встроенным ЦВЗ)
        :param num_bits: сколько бит извлечь
        :return bits_str: строка из '0' и '1'
        """
        defaults = {}
        args = {**defaults, **args}
        input_image = args["input_image"]
        num_bits = args["num_bits"]
        flat = input_image.ravel()
        extracted = flat[:num_bits] & 1
        bits_str = "".join(str(b) for b in extracted)
        return bits_str
