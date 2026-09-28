"""
Гибридный метод DWT-DCT-SVD — встраивание в сингулярные числа
низкочастотной подматрицы DCT-коэффициентов LL-подполосы после дискретного
вейвлет-преобразования блоков.
"""

import numpy as np
cimport numpy as cnp

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    dwt_forward, dwt_inverse, dwt_shape, dwt_integer,
    dwt_rgb, dwt_bits, dwt_luminance, dwt_merge_region,
    init_dct_matrix, apply_dct_8x8, apply_idct_8x8, DBlock,
    power_iteration, power_iteration_sigma,
    qim_embed, qim_extract,
    N_SVD, NN_SVD
)

cnp.import_array()

cdef void _embed_core(double[:, ::1] img_view, int blocks_h, int blocks_w,
                      const cnp.uint8_t[::1] watermark, int block_size, double delta):
    """
    Встраивает биты QIM в сингулярные числа подматриц DCT после DWT.

    Args:
        img_view: изменяемая C-contiguous яркостная матрица double формы (H, W).
        blocks_h: число полных блоков 16x16 по высоте изображения.
        blocks_w: число полных блоков 16x16 по ширине изображения.
        watermark: проверенный одномерный массив uint8 с битами 0 и 1; длина не превышает число блоков.
        block_size: проверенный размер блока 16; DWT выполняется на одном уровне с общим фильтром db5.
        delta: шаг QIM для старшего сингулярного числа подматрицы 4x4.
    """
    cdef double[:, ::1] block = np.empty((block_size, block_size), dtype=np.float64)
    cdef double[:, ::1] coeffs
    cdef DBlock dct_in, dct_out, idct_out
    cdef double svd_matrix[NN_SVD]
    cdef double u[N_SVD]
    cdef double v[N_SVD]
    cdef double s1, s_new, delta_sigma

    cdef int b_idx = 0
    cdef int wm_len = watermark.shape[0]
    cdef int bi, bj, i, j, r, c

    for bi in range(blocks_h):
        for bj in range(blocks_w):
            if b_idx >= wm_len:
                break

            for r in range(block_size):
                for c in range(block_size):
                    block[r, c] = img_view[bi * block_size + r, bj * block_size + c]

            coeffs = dwt_forward(block, 1)

            for i in range(8):
                for j in range(8):
                    dct_in[i][j] = coeffs[i, j]

            apply_dct_8x8(dct_in, dct_out)

            for i in range(N_SVD):
                for j in range(N_SVD):
                    svd_matrix[i * N_SVD + j] = dct_out[i][j]

            power_iteration(svd_matrix, u, v, &s1)

            if s1 < 1e-12:
                b_idx += 1
                continue

            s_new = qim_embed(s1, watermark[b_idx], delta)
            delta_sigma = s_new - s1

            for i in range(N_SVD):
                for j in range(N_SVD):
                    dct_out[i][j] += delta_sigma * u[i] * v[j]

            apply_idct_8x8(dct_out, idct_out)

            for i in range(8):
                for j in range(8):
                    coeffs[i, j] = idct_out[i][j]

            block = dwt_inverse(coeffs, 1)

            for r in range(block_size):
                for c in range(block_size):
                    img_view[bi * block_size + r, bj * block_size + c] = block[r, c]

            b_idx += 1
        if b_idx >= wm_len:
            break


cdef void _extract_core(const double[:, ::1] img_view, int blocks_h, int blocks_w,
                        int[::1] extracted, int wm_length, int block_size, double delta):
    """
    Извлекает биты QIM из сингулярных чисел подматриц DCT после DWT.

    Args:
        img_view: входная C-contiguous яркостная матрица double формы (H, W); не изменяется.
        blocks_h: число полных блоков 16x16 по высоте изображения.
        blocks_w: число полных блоков 16x16 по ширине изображения.
        extracted: изменяемый непрерывный массив int длины не меньше wm_length для извлечённых битов.
        wm_length: положительное число извлекаемых битов, не превышающее число блоков.
        block_size: проверенный размер блока 16; DWT выполняется на одном уровне с общим фильтром db5.
        delta: шаг QIM для старшего сингулярного числа подматрицы 4x4.
    """
    cdef double[:, ::1] block = np.empty((block_size, block_size), dtype=np.float64)
    cdef double[:, ::1] coeffs
    cdef DBlock dct_in, dct_out
    cdef double svd_matrix[NN_SVD]
    cdef double s1

    cdef int b_idx = 0
    cdef int bi, bj, i, j, r, c

    for bi in range(blocks_h):
        for bj in range(blocks_w):
            if b_idx >= wm_length:
                break

            for r in range(block_size):
                for c in range(block_size):
                    block[r, c] = img_view[bi * block_size + r, bj * block_size + c]

            coeffs = dwt_forward(block, 1)

            for i in range(8):
                for j in range(8):
                    dct_in[i][j] = coeffs[i, j]

            apply_dct_8x8(dct_in, dct_out)

            for i in range(N_SVD):
                for j in range(N_SVD):
                    svd_matrix[i * N_SVD + j] = dct_out[i][j]

            s1 = power_iteration_sigma(svd_matrix)
            extracted[b_idx] = qim_extract(s1, delta)

            b_idx += 1
        if b_idx >= wm_length:
            break


class DWTDCT_SVD(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        defaults = {
            "input_image": None,
            "watermark_bits": None,
            "block_size": 16,
            "delta": 80.0,
            "wavelet_name": 'db5',
        }
        args = {**defaults, **args}
        cdef cnp.ndarray[cnp.uint8_t, ndim=3, mode='c'] rgb_c = dwt_rgb(args["input_image"])
        cdef int block_size = dwt_integer("block_size", args["block_size"], 16, 16)
        if args["wavelet_name"] != "db5":
            raise ValueError("wavelet_name must be 'db5': hybrids use the shared 10-tap DWT")
        dwt_shape(block_size, block_size, 1)

        cdef double delta = args["delta"]
        cdef int blocks_h = rgb_c.shape[0] // block_size
        cdef int blocks_w = rgb_c.shape[1] // block_size
        cdef Py_ssize_t capacity = <Py_ssize_t>blocks_h * blocks_w
        cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode='c'] wm_c = dwt_bits(args["watermark_bits"])
        if wm_c.shape[0] > capacity:
            raise ValueError(f"Not enough capacity: needed {wm_c.shape[0]} blocks, available {capacity}.")
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] input_y = dwt_luminance(rgb_c)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] watermarked_y = input_y.copy()
        init_dct_matrix()

        _embed_core(watermarked_y, blocks_h, blocks_w, wm_c, block_size,
                    delta)
        return dwt_merge_region(
            rgb_c, input_y, watermarked_y,
            {"effective_shape": (rgb_c.shape[0], rgb_c.shape[1])},
        )

    @staticmethod
    def extraction(**args):
        defaults = {
            "input_image": None,
            "num_bits": 0,
            "block_size": 16,
            "delta": 80.0,
            "wavelet_name": 'db5',
        }
        args = {**defaults, **args}
        cdef cnp.ndarray[cnp.uint8_t, ndim=3, mode='c'] rgb_c = dwt_rgb(args["input_image"])
        cdef int block_size = dwt_integer("block_size", args["block_size"], 16, 16)
        if args["wavelet_name"] != "db5":
            raise ValueError("wavelet_name must be 'db5': hybrids use the shared 10-tap DWT")
        dwt_shape(block_size, block_size, 1)

        cdef double delta = args["delta"]
        cdef int blocks_h = rgb_c.shape[0] // block_size
        cdef int blocks_w = rgb_c.shape[1] // block_size
        cdef Py_ssize_t capacity = <Py_ssize_t>blocks_h * blocks_w
        cdef int wm_length = dwt_integer("num_bits", args["num_bits"], 1, 2147483647)
        if wm_length > capacity:
            raise ValueError(f"Not enough capacity: needed {wm_length} blocks, available {capacity}.")
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] input_y = dwt_luminance(rgb_c)
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] extracted_wm = np.empty(wm_length, dtype=np.int32)
        init_dct_matrix()

        _extract_core(input_y, blocks_h, blocks_w, extracted_wm, wm_length, block_size,
                    delta)
        if np.any((extracted_wm != 0) & (extracted_wm != 1)):
            raise RuntimeError("dwt_dct_svd extraction produced a non-binary watermark")
        return np.ascontiguousarray(extracted_wm, dtype=np.int8)
