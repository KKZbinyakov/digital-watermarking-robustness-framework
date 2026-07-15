from src.dwarf.ready_solutions.utils.utils import *

class Ready_Spatial_Embeddings:
    """
    Класс, содержащий в себе все готовые пространственные решения встраиваний.
    """
    def embed_lsb(image_path, watermark_bits, output_path):
        """
        Встраивает биты ЦВЗ в младшие биты всех каналов RGB изображения.
        :param image_path: путь к исходному изображению
        :param watermark_bits: строка из '0' и '1' (например, "101100...")
        :param output_path: путь для сохранения результата
        """
        img = Image.open(image_path).convert('RGB')
        data = np.array(img)  # форма (H, W, 3)
        flat = data.ravel()   # одномерный массив всех каналов

        bits = np.array(list(watermark_bits), dtype=np.uint8)
        max_len = len(flat)
        if len(bits) > max_len:
            bits = bits[:max_len]
            print(f"Предупреждение: ЦВЗ слишком длинный, обрезан до {max_len} бит")

        flat[:len(bits)] = (flat[:len(bits)] & 0xFE) | bits
        data = flat.reshape(data.shape)
        result = Image.fromarray(data, mode='RGB')
        result.save(output_path)
        #опционально можно сразу ретурнить, но пока так

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