"""
S. Wu, Z. Wu, M. Huang, D. Shen
A Robust Image Watermarking Scheme Based on Image Normalization and Contourlet Transform (2023)
https://doi.org/10.1109/ACCESS.2023.3319669
"""

import numpy as np
cimport numpy as cnp

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Quantization_Based_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    contourlet_validate_image,
    contourlet_validate_bits,
    contourlet_positive_float,
    contourlet_integer,
    contourlet_boolean,
    contourlet_choice,
    contourlet_to_luma,
    contourlet_lowpass_analysis,
    contourlet_lowpass_synthesis_delta,
    contourlet_embed_core,
    contourlet_extract_core,
    contourlet_arnold,
    contourlet_warp_forward,
    contourlet_warp_inverse_delta,
    contourlet_roi_origin,
    contourlet_centered_roi_origin,
    contourlet_normalized_luma,
    contourlet_profile_geometry,
)

cnp.import_array()


cdef void _embed_core(cnp.ndarray image, const cnp.uint8_t[::1] watermark,
                      double delta0, object roi_size, double centroid_radius,
                      object block_size, double coefficient_scale,
                      object quantization_iterations, object arnold_iterations,
                      object arnold_matrix, bint normalize, double normalization_std,
                      object normalization_bounds, object normalization_sign_mode,
                      object canonical_shape, object compensation_iterations,
                      cnp.ndarray result):
    """
    Встраивает биты ЦВЗ в сингулярные значения блоков Contourlet-коэффициентов.

    Args:
        image: входное RGB-изображение uint8 формы (H, W, 3).
        watermark: проверенный непрерывный массив uint8 с битами 0 и 1; длина не превышает ёмкость.
        delta0: положительный базовый шаг адаптивного квантования.
        roi_size: сторона квадратной области встраивания.
        centroid_radius: положительный радиус поиска яркостного центра области.
        block_size: сторона блока низкочастотных коэффициентов.
        coefficient_scale: положительный масштаб низкочастотных коэффициентов.
        quantization_iterations: число проходов адаптивного квантования.
        arnold_iterations: число итераций перемешивания битов.
        arnold_matrix: целочисленные параметры матрицы преобразования Arnold.
        normalize: флаг моментной нормализации изображения.
        normalization_std: положительный масштаб нормализованных координат.
        normalization_bounds: правило определения границ нормализованного растра.
        normalization_sign_mode: правило выбора знаков нормализации.
        canonical_shape: размеры канонического растра (H, W).
        compensation_iterations: число проходов компенсации округления пикселей.
        result: выходной непрерывный RGB-массив uint8 той же формы, что image.
    """
    payload = contourlet_arnold(np.asarray(watermark), arnold_iterations, False, arnold_matrix)
    normalized, transform = contourlet_normalized_luma(
        image,
        normalize,
        normalization_std,
        canonical_shape,
        bounds_mode=normalization_bounds,
        sign_mode=normalization_sign_mode,
    )
    if min(normalized.shape) < roi_size:
        raise ValueError(f"roi_size={roi_size} does not fit normalized image shape={normalized.shape}")
    top, left = (
        contourlet_roi_origin(normalized, roi_size, centroid_radius)
        if normalize
        else contourlet_centered_roi_origin(normalized, roi_size)
    )
    roi = normalized[top : top + roi_size, left : left + roi_size]
    lowpass = contourlet_lowpass_analysis(roi, coefficient_scale)
    modified_lowpass = contourlet_embed_core(
        lowpass,
        payload,
        delta0,
        block_size,
        max_iterations=quantization_iterations,
    )
    target = normalized.copy()
    target[top : top + roi_size, left : left + roi_size] = roi + contourlet_lowpass_synthesis_delta(
        modified_lowpass - lowpass,
        coefficient_scale,
    )

    normalized_delta = target - normalized
    correction = (
        contourlet_warp_inverse_delta(normalized_delta, transform, (image.shape[0], image.shape[1]))
        if normalize else normalized_delta
    )
    pixels = np.clip(image.astype(np.float64) + correction[..., None], 0.0, 255.0)

    for _ in range(compensation_iterations):
        current = np.floor(np.clip(pixels, 0.0, 255.0) + 0.5).astype(np.uint8)
        current_luma = contourlet_to_luma(current)
        current_normalized = contourlet_warp_forward(current_luma, transform, normalized.shape) if normalize else current_luma
        residual = target - current_normalized
        correction = (
            contourlet_warp_inverse_delta(residual, transform, (image.shape[0], image.shape[1]))
            if normalize else residual
        )
        pixels += correction[..., None]
        pixels = np.clip(pixels, 0.0, 255.0)

    result[:] = np.floor(pixels + 0.5).astype(np.uint8)


cdef void _extract_core(cnp.ndarray image, double delta0, object roi_size,
                        double centroid_radius, object block_size,
                        double coefficient_scale, object arnold_iterations,
                        object arnold_matrix, bint normalize, double normalization_std,
                        object normalization_bounds, object normalization_sign_mode,
                        object canonical_shape, cnp.int8_t[::1] extracted):
    """
    Извлекает биты ЦВЗ по чётности квантованных сингулярных значений.

    Args:
        image: входное RGB-изображение uint8 формы (H, W, 3).
        delta0: положительный базовый шаг адаптивного квантования.
        roi_size: сторона квадратной области извлечения.
        centroid_radius: положительный радиус поиска яркостного центра области.
        block_size: сторона блока низкочастотных коэффициентов.
        coefficient_scale: положительный масштаб низкочастотных коэффициентов.
        arnold_iterations: число итераций обратного перемешивания битов.
        arnold_matrix: целочисленные параметры матрицы преобразования Arnold.
        normalize: флаг моментной нормализации изображения.
        normalization_std: положительный масштаб нормализованных координат.
        normalization_bounds: правило определения границ нормализованного растра.
        normalization_sign_mode: правило выбора знаков нормализации.
        canonical_shape: размеры канонического растра (H, W).
        extracted: выходной непрерывный массив int8; его длина задаёт число битов и не превышает ёмкость.
    """
    cdef const cnp.int8_t[::1] payload
    cdef Py_ssize_t index

    normalized, _ = contourlet_normalized_luma(
        image,
        normalize,
        normalization_std,
        canonical_shape,
        bounds_mode=normalization_bounds,
        sign_mode=normalization_sign_mode,
    )
    if min(normalized.shape) < roi_size:
        raise ValueError(f"roi_size={roi_size} does not fit normalized image shape={normalized.shape}")
    top, left = (
        contourlet_roi_origin(normalized, roi_size, centroid_radius)
        if normalize
        else contourlet_centered_roi_origin(normalized, roi_size)
    )
    roi = normalized[top : top + roi_size, left : left + roi_size]
    decoded = contourlet_extract_core(
        contourlet_lowpass_analysis(roi, coefficient_scale),
        extracted.shape[0],
        delta0,
        block_size,
    )
    payload = contourlet_arnold(decoded, arnold_iterations, inverse=True, matrix=arnold_matrix).astype(np.int8, copy=False)
    for index in range(extracted.shape[0]):
        extracted[index] = payload[index]


class Contourlet(Ready_Quantization_Based_Embeddings):
    @staticmethod
    def embedding(**args):
        defaults = {
            "input_image": None,
            "watermark_bits": None,
            "delta0": 48.0,
            "roi_size": 256,
            "centroid_radius": 512.0,
            "block_size": 4,
            "coefficient_scale": 4.710,
            "quantization_iterations": 6,
            "arnold_iterations": 5,
            "arnold_a": 1,
            "arnold_b": 1,
            "arnold_c": 1,
            "arnold_d": 2,
            "normalize": True,
            "normalization_std": 256.0,
            "normalization_bounds": "full",
            "normalization_sign_mode": "positive",
            "canonical_size": None,
            "canonical_height": 512,
            "canonical_width": 512,
            "compensation_iterations": 3,
        }
        args = {**defaults, **{name: args[name] for name in defaults.keys() & args.keys()}}

        cdef cnp.ndarray image = contourlet_validate_image(args["input_image"])
        watermark = contourlet_validate_bits(args["watermark_bits"])
        cdef double delta0 = contourlet_positive_float("delta0", args["delta0"])
        roi_size = contourlet_integer("roi_size", args["roi_size"], 8)
        cdef double centroid_radius = contourlet_positive_float("centroid_radius", args["centroid_radius"])
        block_size = contourlet_integer("block_size", args["block_size"], 1)
        cdef double coefficient_scale = contourlet_positive_float("coefficient_scale", args["coefficient_scale"])
        quantization_iterations = contourlet_integer("quantization_iterations", args["quantization_iterations"], 1)
        arnold_iterations = contourlet_integer("arnold_iterations", args["arnold_iterations"])
        arnold_matrix = tuple(contourlet_integer(f"arnold_{name}", args[f"arnold_{name}"]) for name in "abcd")
        cdef double normalization_std = contourlet_positive_float("normalization_std", args["normalization_std"])
        normalization_bounds = contourlet_choice("normalization_bounds", args["normalization_bounds"], ("fixed", "full"))
        normalization_sign_mode = contourlet_choice(
            "normalization_sign_mode",
            args["normalization_sign_mode"],
            ("fifth_moment", "positive"),
        )
        if args["canonical_size"] is None:
            canonical_shape = (
                contourlet_integer("canonical_height", args["canonical_height"], roi_size),
                contourlet_integer("canonical_width", args["canonical_width"], roi_size),
            )
        else:
            canonical_size = contourlet_integer("canonical_size", args["canonical_size"], roi_size)
            canonical_shape = (canonical_size, canonical_size)
        compensation_iterations = contourlet_integer("compensation_iterations", args["compensation_iterations"])
        cdef bint normalize = contourlet_boolean("normalize", args["normalize"])
        contourlet_profile_geometry(roi_size, block_size, canonical_shape)

        cdef Py_ssize_t capacity = (roi_size // (2 * block_size)) ** 2
        cdef Py_ssize_t count = watermark.size
        if count > capacity:
            raise ValueError(f"Not enough payload capacity: need {count} bits, available {capacity}")

        cdef cnp.ndarray wm = np.ascontiguousarray(watermark)
        cdef cnp.ndarray result = image.copy()
        _embed_core(
            image,
            wm,
            delta0,
            roi_size,
            centroid_radius,
            block_size,
            coefficient_scale,
            quantization_iterations,
            arnold_iterations,
            arnold_matrix,
            normalize,
            normalization_std,
            normalization_bounds,
            normalization_sign_mode,
            canonical_shape,
            compensation_iterations,
            result,
        )

        return result

    @staticmethod
    def extraction(**args):
        defaults = {
            "input_image": None,
            "num_bits": None,
            "delta0": 48.0,
            "roi_size": 256,
            "centroid_radius": 512.0,
            "block_size": 4,
            "coefficient_scale": 4.710,
            "arnold_iterations": 5,
            "arnold_a": 1,
            "arnold_b": 1,
            "arnold_c": 1,
            "arnold_d": 2,
            "normalize": True,
            "normalization_std": 256.0,
            "normalization_bounds": "full",
            "normalization_sign_mode": "positive",
            "canonical_size": None,
            "canonical_height": 512,
            "canonical_width": 512,
        }
        args = {**defaults, **{name: args[name] for name in defaults.keys() & args.keys()}}

        cdef cnp.ndarray image = contourlet_validate_image(args["input_image"])
        num_bits = contourlet_integer("num_bits", args["num_bits"], 1)
        cdef double delta0 = contourlet_positive_float("delta0", args["delta0"])
        roi_size = contourlet_integer("roi_size", args["roi_size"], 8)
        cdef double centroid_radius = contourlet_positive_float("centroid_radius", args["centroid_radius"])
        block_size = contourlet_integer("block_size", args["block_size"], 1)
        cdef double coefficient_scale = contourlet_positive_float("coefficient_scale", args["coefficient_scale"])
        arnold_iterations = contourlet_integer("arnold_iterations", args["arnold_iterations"])
        arnold_matrix = tuple(contourlet_integer(f"arnold_{name}", args[f"arnold_{name}"]) for name in "abcd")
        cdef double normalization_std = contourlet_positive_float("normalization_std", args["normalization_std"])
        normalization_bounds = contourlet_choice("normalization_bounds", args["normalization_bounds"], ("fixed", "full"))
        normalization_sign_mode = contourlet_choice(
            "normalization_sign_mode",
            args["normalization_sign_mode"],
            ("fifth_moment", "positive"),
        )
        if args["canonical_size"] is None:
            canonical_shape = (
                contourlet_integer("canonical_height", args["canonical_height"], roi_size),
                contourlet_integer("canonical_width", args["canonical_width"], roi_size),
            )
        else:
            canonical_size = contourlet_integer("canonical_size", args["canonical_size"], roi_size)
            canonical_shape = (canonical_size, canonical_size)
        cdef bint normalize = contourlet_boolean("normalize", args["normalize"])
        contourlet_profile_geometry(roi_size, block_size, canonical_shape)

        cdef Py_ssize_t capacity = (roi_size // (2 * block_size)) ** 2
        if num_bits > capacity:
            raise ValueError(f"Not enough payload capacity: need {num_bits} bits, available {capacity}")

        cdef cnp.ndarray extracted = np.empty(num_bits, dtype=np.int8)
        _extract_core(
            image,
            delta0,
            roi_size,
            centroid_radius,
            block_size,
            coefficient_scale,
            arnold_iterations,
            arnold_matrix,
            normalize,
            normalization_std,
            normalization_bounds,
            normalization_sign_mode,
            canonical_shape,
            extracted,
        )
        return extracted
