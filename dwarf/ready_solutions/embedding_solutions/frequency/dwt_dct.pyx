"""
Гибридный метод DWT-DCT — встраивание в DCT-коэффициенты LL-подполосы
после дискретного вейвлет-преобразования блоков.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    get_wavelet_filters, dwt_2d_block, idwt_2d_block,
    init_dct_matrix, apply_dct_8x8, apply_idct_8x8, DBlock
)

cnp.import_array()

cdef void _embed_core(double[:, :] img_view, int blocks_h, int blocks_w,
                      int[:] watermark, int block_size, double margin, double threshold,
                      double[:, :] block,
                      double[:, :] LL, double[:, :] LH, double[:, :] HL, double[:, :] HH,
                      double* temp_ptr, double* row_a_ptr, double* row_d_ptr,
                      double* col_in_ptr, double* col_a_ptr, double* col_d_ptr,
                      double* col_out_ptr, double* row_out_ptr,
                      const double* h, const double* g, int L):
    """
    Ядро встраивания ЦВЗ.

    Args:
        img_view: представление изображения для записи.
        blocks_h: количество блоков по высоте.
        blocks_w: количество блоков по ширине.
        watermark: ЦВЗ.
        block_size: размер блока.
        margin: величина модификации коэффициентов.
        threshold: порог для определения бита.
        block: буфер блока.
        LL, LH, HL, HH: буферы подполос.
        temp_ptr: временный буфер.
        row_a_ptr, row_d_ptr: буферы для строк.
        col_in_ptr, col_a_ptr, col_d_ptr: буферы для столбцов.
        col_out_ptr, row_out_ptr: выходные буферы.
        h, g: фильтры вейвлета.
        L: длина фильтра.
    """
    cdef DBlock dct_in, dct_out, idct_out
    cdef int b_idx = 0
    cdef int wm_len = watermark.shape[0]
    cdef int bi, bj, i, j, r, c
    cdef double c1, c2, k

    for bi in range(blocks_h):
        for bj in range(blocks_w):
            if b_idx >= wm_len:
                break

            for r in range(block_size):
                for c in range(block_size):
                    block[r, c] = img_view[bi * block_size + r, bj * block_size + c]

            dwt_2d_block(block, LL, LH, HL, HH,
                        temp_ptr, row_a_ptr, row_d_ptr,
                        col_in_ptr, col_a_ptr, col_d_ptr,
                        h, g, L, block_size)

            for i in range(8):
                for j in range(8):
                    dct_in[i][j] = LL[i, j]

            apply_dct_8x8(dct_in, dct_out)

            c1 = dct_out[3][4]
            c2 = dct_out[4][3]
            k = fabs(c1) - fabs(c2)

            if watermark[b_idx] == 1:
                if k <= threshold:
                    dct_out[3][4] = fabs(c2) + margin if c1 >= 0 else -(fabs(c2) + margin)
            else:
                if k >= -threshold:
                    dct_out[4][3] = fabs(c1) + margin if c2 >= 0 else -(fabs(c1) + margin)

            apply_idct_8x8(dct_out, idct_out)

            for i in range(8):
                for j in range(8):
                    LL[i, j] = idct_out[i][j]

            idwt_2d_block(LL, LH, HL, HH, block,
                         temp_ptr, col_a_ptr, col_d_ptr, col_out_ptr,
                         row_a_ptr, row_d_ptr, row_out_ptr,
                         h, g, L, block_size)

            for r in range(block_size):
                for c in range(block_size):
                    img_view[bi * block_size + r, bj * block_size + c] = block[r, c]

            b_idx += 1
        if b_idx >= wm_len:
            break

cdef void _extract_core(double[:, :] img_view, int blocks_h, int blocks_w,
                        int[:] extracted, int wm_length, int block_size, double threshold,
                        double[:, :] block,
                        double[:, :] LL, double[:, :] LH, double[:, :] HL, double[:, :] HH,
                        double* temp_ptr, double* row_a_ptr, double* row_d_ptr,
                        double* col_in_ptr, double* col_a_ptr, double* col_d_ptr,
                        double* col_out_ptr, double* row_out_ptr,
                        const double* h, const double* g, int L):
    """
    Ядро извлечения ЦВЗ.

    Args:
        img_view: представление изображения с ЦВЗ.
        blocks_h: количество блоков по высоте.
        blocks_w: количество блоков по ширине.
        extracted: выходной массив для извлечённых бит.
        wm_length: длина ЦВЗ.
        block_size: размер блока.
        threshold: порог для определения бита.
        block: буфер блока.
        LL, LH, HL, HH: буферы подполос.
        temp_ptr: временный буфер.
        row_a_ptr, row_d_ptr: буферы для строк.
        col_in_ptr, col_a_ptr, col_d_ptr: буферы для столбцов.
        col_out_ptr, row_out_ptr: выходные буферы.
        h, g: фильтры вейвлета.
        L: длина фильтра.
    """
    cdef DBlock dct_in, dct_out
    cdef int b_idx = 0
    cdef int bi, bj, i, j, r, c
    cdef double c1, c2, k

    for bi in range(blocks_h):
        for bj in range(blocks_w):
            if b_idx >= wm_length:
                break

            for r in range(block_size):
                for c in range(block_size):
                    block[r, c] = img_view[bi * block_size + r, bj * block_size + c]

            dwt_2d_block(block, LL, LH, HL, HH,
                        temp_ptr, row_a_ptr, row_d_ptr,
                        col_in_ptr, col_a_ptr, col_d_ptr,
                        h, g, L, block_size)

            for i in range(8):
                for j in range(8):
                    dct_in[i][j] = LL[i, j]

            apply_dct_8x8(dct_in, dct_out)

            c1 = dct_out[3][4]
            c2 = dct_out[4][3]
            k = fabs(c1) - fabs(c2)

            if k >= threshold:
                extracted[b_idx] = 1
            elif k <= -threshold:
                extracted[b_idx] = 0
            else:
                extracted[b_idx] = -1

            b_idx += 1
        if b_idx >= wm_length:
            break

class DWTDCT(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        """
        Встраивает биты ЦВЗ в DCT-коэффициенты LL-подполосы после DWT.

        :param input_image: матрица входного изображения (канал яркости Y).
        :param watermark_bits: массив битов ЦВЗ.
        :param block_size: размер блока для DWT (должен быть равен 16).
        :param margin: величина модификации коэффициентов.
        :param threshold: порог для определения бита.
        :param wavelet_name: тип вейвлета (haar, db4, sym4).
        :return output_image: матрица изображения с встроенным ЦВЗ.
        """
        defaults = {
                    "input_image": None,
                    "watermark_bits": None,
                    "block_size": 16,
                    "margin": 150.0,
                    "threshold": 25.0,
                    "wavelet_name": "haar"
                }
        args = {**defaults, **args}
        image = args.get["input_image"]
        watermark = args.get["watermark_bits"]
        if image is None or watermark is None:
            raise ValueError("input_image/image_path or watermark_bits not given")

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(image, dtype=np.float64)
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] wm_c = np.ascontiguousarray(watermark, dtype=np.int32)
        cdef int block_size = int(args["block_size"])
        cdef double margin = args["margin"]
        cdef double threshold = args["threshold"]
        cdef bytes wavelet_name = args["wavelet_name"].encode('utf-8')

        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]

        if block_size != 16:
            raise ValueError("block_size must be 16")

        cdef int blocks_h = H // block_size
        cdef int blocks_w = W // block_size
        cdef int capacity = blocks_h * blocks_w
        cdef int wm_len = wm_c.shape[0]

        if wm_len > capacity:
            raise ValueError(
                f"Not enough capacity: needed {wm_len} blocks, available {capacity}."
            )

        cdef double h_buf[32]
        cdef double g_buf[32]
        cdef int L_buf
        get_wavelet_filters(wavelet_name, h_buf, g_buf, &L_buf)

        init_dct_matrix()

        cdef int half = block_size // 2

        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] temp_np = np.zeros(block_size * block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_a_np = np.zeros(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_d_np = np.zeros(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_in_np = np.zeros(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_a_np = np.zeros(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_d_np = np.zeros(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_out_np = np.zeros(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_out_np = np.zeros(block_size, dtype=np.float64)

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] block_np = np.zeros((block_size, block_size), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] LL_np = np.zeros((half, half), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] LH_np = np.zeros((half, half), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] HL_np = np.zeros((half, half), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] HH_np = np.zeros((half, half), dtype=np.float64)

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] watermarked_img = img_c.copy()
        cdef double[:, :] img_view = watermarked_img

        _embed_core(img_view, blocks_h, blocks_w, wm_c, block_size, margin, threshold,
                   block_np, LL_np, LH_np, HL_np, HH_np,
                   &temp_np[0], &row_a_np[0], &row_d_np[0],
                   &col_in_np[0], &col_a_np[0], &col_d_np[0],
                   &col_out_np[0], &row_out_np[0],
                   h_buf, g_buf, L_buf)

        return watermarked_img

    @staticmethod
    def extraction(**args):
        """
        Извлекает биты ЦВЗ из DCT-коэффициентов LL-подполосы после DWT.

        :param input_image: матрица изображения с ЦВЗ (канал яркости Y).
        :param num_bits: длина ЦВЗ.
        :param block_size: размер блока для DWT.
        :param threshold: порог для определения бита.
        :param wavelet_name: тип вейвлета.
        :return extracted_wm: извлечённый ЦВЗ.
        """
        defaults = {
                    "input_image": None,
                    "num_bits": 0,
                    "block_size": 16,
                    "threshold": 25.0,
                    "wavelet_name": "haar"
                }
        args = {**defaults, **args}
        image = args.get["input_image"]
        num_bits = args.get["num_bits"]
        if image is None or not num_bits:
            raise ValueError("input_image/image_path or watermark_bits not given")

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(image, dtype=np.float64)
        cdef int wm_length = num_bits
        cdef int block_size = int(args["block_size"])
        cdef double threshold = args["threshold"]
        cdef bytes wavelet_name = args["wavelet_name"].encode('utf-8')

        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]

        if block_size != 16:
            raise ValueError("block_size must be 16")

        cdef int blocks_h = H // block_size
        cdef int blocks_w = W // block_size
        cdef int capacity = blocks_h * blocks_w

        if wm_length > capacity:
            raise ValueError(f"wm_length is more than the number of blocks.")

        cdef double h_buf[32]
        cdef double g_buf[32]
        cdef int L_buf
        get_wavelet_filters(wavelet_name, h_buf, g_buf, &L_buf)

        init_dct_matrix()

        cdef int half = block_size // 2

        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] extracted_wm = np.zeros(wm_length, dtype=np.int32)
        cdef int[:] extracted = extracted_wm

        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] temp_np = np.zeros(block_size * block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_a_np = np.zeros(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_d_np = np.zeros(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_in_np = np.zeros(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_a_np = np.zeros(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_d_np = np.zeros(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_out_np = np.zeros(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_out_np = np.zeros(block_size, dtype=np.float64)

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] block_np = np.zeros((block_size, block_size), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] LL_np = np.zeros((half, half), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] LH_np = np.zeros((half, half), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] HL_np = np.zeros((half, half), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] HH_np = np.zeros((half, half), dtype=np.float64)

        cdef double[:, :] img_view = img_c

        _extract_core(img_view, blocks_h, blocks_w, extracted, wm_length, block_size, threshold,
                    block_np, LL_np, LH_np, HL_np, HH_np,
                    &temp_np[0], &row_a_np[0], &row_d_np[0],
                    &col_in_np[0], &col_a_np[0], &col_d_np[0],
                    &col_out_np[0], &row_out_np[0],
                    h_buf, g_buf, L_buf)

        return extracted_wm