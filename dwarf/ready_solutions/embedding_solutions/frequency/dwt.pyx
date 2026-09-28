"""
Deepa Kundur and Dimitrios Hatzinakos
Digital Watermarking Using Multiresolution Wavelet Decomposition (1998)
https://www.comm.utoronto.ca/dkundur/pub_pdfs/KunHatICASSP98.pdf
"""
from sys import maxsize

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx import DWT_PROFILE as PROFILE
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    dwt_integer, dwt_parameters, dwt_rgb, dwt_bits,
    dwt_check_capacity, dwt_image_region, dwt_luminance, dwt_merge_region,
    dwt_forward, dwt_inverse, dwt_embed_coefficients_inplace,
    dwt_decode_coefficients, dwt_require_decoded_bits, dwt_capacity,
)


cdef object _embed_core(object args):
    """
    Выполняет встраивание ЦВЗ в многоуровневые коэффициенты DWT.

    Args:
        args: словарь с input_image (RGB uint8), watermark_bits (бинарный uint8) и параметрами levels, q,
            repetitions, key_seed, span_epsilon, geometry_mode.

    Returns:
        output_image: новое RGB-изображение uint8 исходной формы; пиксели вне рабочей области сохранены.
    """
    parameters = dict(args)
    image = dwt_rgb(parameters.pop("input_image", None))
    bits = dwt_bits(parameters.pop("watermark_bits", None))
    config = dwt_parameters(parameters)
    region, geometry = dwt_image_region(image, config)
    dwt_check_capacity(region.shape, bits.size, config)
    luminance = dwt_luminance(region)
    coefficients = dwt_forward(luminance, config["levels"])
    dwt_embed_coefficients_inplace(
        coefficients, bits, config["levels"], config["q"],
        config["repetitions"], config["key_seed"], config["span_epsilon"],
    )
    modified = dwt_inverse(coefficients, config["levels"])
    return dwt_merge_region(image, luminance, modified, geometry)


cdef object _extract_core(object args):
    """
    Извлекает биты DWT и сведения о голосовании.

    Args:
        args: словарь с input_image (RGB uint8), положительным num_bits, параметрами встраивания и
            extraction_mode (all или coarsest).

    Returns:
        details: словарь с bits (int8: 0, 1 или -1 без голосов), голосами, стираниями, геометрией и
            capacity_bits; отсутствие голосов здесь не вызывает ошибку.
    """
    parameters = dict(args)
    image = dwt_rgb(parameters.pop("input_image", None))
    num_bits = dwt_integer("num_bits", parameters.pop("num_bits", 0), 1, maxsize)
    config = dwt_parameters(parameters, extraction=True)
    region, geometry = dwt_image_region(image, config)
    dwt_check_capacity(region.shape, num_bits, config)
    coefficients = dwt_forward(dwt_luminance(region), config["levels"])
    details = dwt_decode_coefficients(
        coefficients, num_bits, config["levels"], config["q"],
        config["repetitions"], config["key_seed"], config["span_epsilon"],
        config["extraction_mode"] == "coarsest",
    )
    details["geometry"] = geometry
    details["capacity_bits"] = dwt_capacity(
        image.shape, config["levels"], config["repetitions"], config["geometry_mode"],
    )
    return details


class DWT(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        return _embed_core(args)

    @staticmethod
    def extraction(**args):
        return dwt_require_decoded_bits(_extract_core(args))

    @staticmethod
    def diagnostics(**args):
        return _extract_core(args)

    @staticmethod
    def capacity(image_shape, levels=4, repetitions=1, geometry_mode="roi"):
        return dwt_capacity(image_shape, levels, repetitions, geometry_mode)
