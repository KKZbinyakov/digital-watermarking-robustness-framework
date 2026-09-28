"""
Brian Chen and Gregory W. Wornell
"Quantization Index Modulation: A Class of Provably Good Methods for Digital Watermarking and Information Embedding (2001)
https://ieeexplore.ieee.org/document/866336
"""

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    contourlet_qim_embed_image,
    contourlet_qim_extract_image,
    contourlet_qim_options,
)


CONTOURLET_PROFILE_ID = 'lowpass-qim-cython-v1'

cdef object _embed_core(object input_image, object watermark_bits, object profile):
    """
    Встраивает ЦВЗ в низкочастотную ветвь и формирует uint8-изображение.

    Args:
        input_image: RGB-массив uint8 формы (H, W, 3) либо полутоновый массив (H, W).
        watermark_bits: одномерный целочисленный или булев массив с битами 0 и 1.
        profile: проверенный профиль QIM с levels, delta, channel, key и repetitions.

    Returns:
        output_image: новый массив uint8 исходной формы после ограничения диапазона и округления.
    """
    return contourlet_qim_embed_image(input_image, watermark_bits, profile)


cdef object _extract_core(object input_image, object num_bits, object profile):
    """
    Выполняет слепое извлечение ЦВЗ из низкочастотной ветви.

    Args:
        input_image: принятое RGB- или полутоновое изображение uint8 без доступа к оригиналу.
        num_bits: неотрицательная длина извлекаемого сообщения в битах.
        profile: профиль QIM с теми же levels, delta, channel, key и repetitions, что при встраивании.

    Returns:
        bits: непрерывный одномерный массив int8 длины num_bits со значениями 0 и 1.
    """
    return contourlet_qim_extract_image(input_image, num_bits, profile)


class Contourlet(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**kwargs):
        defaults = {
            'input_image': None,
            'watermark_bits': None,
            'levels': 2,
            'delta': 32.0,
            'channel': 1,
            'key': 0,
            'repetitions': 3,
            'seed': None,
        }
        input_image, watermark_bits, profile = contourlet_qim_options(
            kwargs, defaults, 'watermark_bits'
        )
        return _embed_core(input_image, watermark_bits, profile)

    @staticmethod
    def extraction(**kwargs):
        defaults = {
            'input_image': None,
            'num_bits': None,
            'levels': 2,
            'delta': 32.0,
            'channel': 1,
            'key': 0,
            'repetitions': 3,
            'seed': None,
        }
        input_image, num_bits, profile = contourlet_qim_options(
            kwargs, defaults, 'num_bits'
        )
        return _extract_core(input_image, num_bits, profile)
