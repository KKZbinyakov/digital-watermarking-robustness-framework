"""Атака перекодирования в TIFF: сжатие без потерь, контрольная для нулевого BER."""

from dwarf.core.attack_orchestrator.attack_core import Ready_Compression_Attacks
from dwarf.ready_solutions.utils.attack_utils import roundtrip_buffer


class Tiff(Ready_Compression_Attacks):
    """
    Атака перекодирования в TIFF с LZW или Deflate.

    Сжатие без потерь, пиксели не меняются
    """

    @staticmethod
    def attack(**args):
        """
        Перекодирует изображение в TIFF.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                compression (str): 'tiff_lzw' или 'tiff_adobe_deflate' (по умолчанию 'tiff_lzw')

        Returns:
            np.ndarray: матрица изображения после атаки
        """
        defaults = {"input_image": None, "compression": "tiff_lzw"}
        args = {**defaults, **args}
        input_image = args["input_image"]
        compression = args["compression"]

        return roundtrip_buffer(input_image, "TIFF", compression=compression)
