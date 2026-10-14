"""
Xiangguang Xiong
An Improved DCT Based Color Image Watermarking Scheme (2016)
https://www.atlantis-press.com/article/25866956.pdf
"""

import operator
import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    DBlock, init_dct_matrix, apply_dct_8x8, apply_idct_8x8,
)

cnp.import_array()


cdef cnp.ndarray _validate_ycbcr_image(object image):
    """Проверяет структуру YCbCr-массива без преобразования его каналов."""
    if image is None:
        raise ValueError("input_image is required")
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"input_image must have shape (H, W, 3) in Y, Cb, Cr order, got {arr.shape}")
    if arr.dtype != np.float64:
        raise TypeError(f"input_image must be YCbCr with dtype float64, got {arr.dtype}")
    if arr.shape[0] < 8 or arr.shape[1] < 8:
        raise ValueError("input_image must contain at least one complete 8x8 block")
    if not np.all(np.isfinite(arr)):
        raise ValueError("input_image must contain only finite YCbCr values")
    return np.ascontiguousarray(arr)


cdef void _embed_core(double[:, :] image_y, Py_ssize_t blocks_w,
                      const cnp.uint8_t[::1] watermark, double strength):
    """
    Встраивает биты ЦВЗ в пары DCT-коэффициентов блоков 8x8.

    Args:
        image_y: изменяемая яркостная матрица double формы (H, W).
        blocks_w: число полных блоков 8x8 в строке изображения.
        watermark: проверенный непрерывный массив uint8 с битами 0 и 1; длина не превышает ёмкость.
        strength: положительная сила встраивания для пары коэффициентов [4, 1] и [3, 2].
    """
    cdef DBlock spatial, coeffs, restored
    cdef Py_ssize_t index, y0, x0
    cdef int r, c, bit
    cdef double a, b, difference, delta
    cdef bint flag

    for index in range(watermark.shape[0]):
        y0 = (index // blocks_w) * 8
        x0 = (index % blocks_w) * 8
        for r in range(8):
            for c in range(8):
                spatial[r][c] = image_y[y0 + r, x0 + c]
        apply_dct_8x8(spatial, coeffs)

        a = coeffs[4][1]
        b = coeffs[3][2]
        flag = a >= b
        bit = watermark[index]
        difference = fabs(a - b)
        delta = (difference + strength) / 2.0

        if not flag and bit == 0:
            if difference < strength:
                a -= delta
                b += delta
        elif flag and bit == 1:
            if difference < strength:
                a += delta
                b -= delta
        elif flag and bit == 0:
            a -= delta
            b += delta
        else:
            a += delta
            b -= delta

        coeffs[4][1] = a
        coeffs[3][2] = b
        apply_idct_8x8(coeffs, restored)
        for r in range(8):
            for c in range(8):
                image_y[y0 + r, x0 + c] = restored[r][c]


cdef void _extract_core(const double[:, :] image_y, Py_ssize_t blocks_w,
                        cnp.int8_t[::1] extracted):
    """
    Извлекает биты ЦВЗ сравнением пар DCT-коэффициентов.

    Args:
        image_y: входная яркостная матрица double формы (H, W).
        blocks_w: число полных блоков 8x8 в строке изображения.
        extracted: выходной непрерывный массив int8; его длина задаёт число битов и не превышает ёмкость.
    """
    cdef DBlock spatial, coeffs
    cdef Py_ssize_t index, y0, x0
    cdef int r, c
    cdef double a, b

    for index in range(extracted.shape[0]):
        y0 = (index // blocks_w) * 8
        x0 = (index % blocks_w) * 8
        for r in range(8):
            for c in range(8):
                spatial[r][c] = image_y[y0 + r, x0 + c]
        apply_dct_8x8(spatial, coeffs)
        a, b = coeffs[4][1], coeffs[3][2]
        if a > b:
            extracted[index] = 1
        elif a < b:
            extracted[index] = 0
        else:
            extracted[index] = -1


class DCT(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        unknown = set(args) - {"input_image", "watermark_bits", "strength"}
        if unknown:
            raise TypeError(f"Unknown Xiong DCT embedding parameters: {sorted(unknown)}")

        cdef cnp.ndarray ycbcr = _validate_ycbcr_image(args.get("input_image"))
        watermark = np.asarray(args.get("watermark_bits"))
        if watermark.ndim != 1:
            raise ValueError("watermark_bits must be a one-dimensional array")
        if watermark.dtype != np.uint8:
            raise TypeError("watermark_bits must have dtype uint8")
        if watermark.size == 0:
            raise ValueError("watermark_bits must not be empty")
        if np.any((watermark != 0) & (watermark != 1)):
            raise ValueError("watermark_bits must contain only 0 and 1")
        cdef double strength = float(args.get("strength", 16.0))
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError("strength (K) must be finite and strictly positive")

        cdef Py_ssize_t blocks_w = ycbcr.shape[1] // 8
        cdef Py_ssize_t capacity = (ycbcr.shape[0] // 8) * blocks_w
        cdef Py_ssize_t count = watermark.size
        if count > capacity:
            raise ValueError(f"Not enough capacity: requested {count} bits, available {capacity}")

        cdef cnp.ndarray wm = np.ascontiguousarray(watermark)
        cdef cnp.ndarray result = ycbcr.copy()
        cdef double[:, :, ::1] channels = result
        init_dct_matrix()
        _embed_core(channels[:, :, 0], blocks_w, wm, strength)

        return result

    @staticmethod
    def extraction(**args):
        unknown = set(args) - {"input_image", "num_bits"}
        if unknown:
            raise TypeError(f"Unknown Xiong DCT extraction parameters: {sorted(unknown)}")
        cdef cnp.ndarray ycbcr = _validate_ycbcr_image(args.get("input_image"))
        value = args.get("num_bits")
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("num_bits must be an integer, not bool")
        try:
            value = operator.index(value)
        except TypeError:
            raise TypeError("num_bits must be an integer") from None
        cdef Py_ssize_t blocks_w = ycbcr.shape[1] // 8
        cdef Py_ssize_t capacity = (ycbcr.shape[0] // 8) * blocks_w
        if not 1 <= value <= capacity:
            raise ValueError(f"num_bits must be in [1, {capacity}], got {value}")
        cdef cnp.ndarray extracted = np.empty(value, dtype=np.int8)
        cdef const double[:, :, ::1] channels = ycbcr
        init_dct_matrix()
        _extract_core(channels[:, :, 0], blocks_w, extracted)
        return extracted
