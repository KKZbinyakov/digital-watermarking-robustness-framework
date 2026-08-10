"""
Гибридный метод DWT-DCT-SVD — встраивание в сингулярные числа
низкочастотной подматрицы DCT-коэффициентов LL-подполосы после дискретного
вейвлет-преобразования блоков.
"""

import numpy as np
cimport numpy as cnp

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pxd cimport (
    get_wavelet_filters, dwt_2d_block, idwt_2d_block,
    init_dct_matrix, apply_dct_8x8, apply_idct_8x8, DBlock,
    power_iteration, power_iteration_sigma,
    qim_embed, qim_extract,
    N_SVD, NN_SVD
)

cnp.import_array()

cdef void _embed_core(double[:, :] img_view, int blocks_h, int blocks_w,
                      int[:] watermark, int block_size, double delta,
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
        delta: шаг квантования.
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

            dwt_2d_block(block, LL, LH, HL, HH,
                        temp_ptr, row_a_ptr, row_d_ptr,
                        col_in_ptr, col_a_ptr, col_d_ptr,
                        h, g, L, block_size)

            for i in range(8):
                for j in range(8):
                    dct_in[i][j] = LL[i, j]

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
                        int[:] extracted, int wm_length, int block_size, double delta,
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
        delta: шаг квантования.
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

            dwt_2d_block(block, LL, LH, HL, HH,
                        temp_ptr, row_a_ptr, row_d_ptr,
                        col_in_ptr, col_a_ptr, col_d_ptr,
                        h, g, L, block_size)

            for i in range(8):
                for j in range(8):
                    dct_in[i][j] = LL[i, j]

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
        """
        Встраивает биты ЦВЗ в сингулярные числа низкочастотной подматрицы
        DCT-коэффициентов LL-подполосы после DWT.

        :param input_image: матрица входного изображения (канал яркости Y).
        :param watermark_bits: массив битов ЦВЗ.
        :param block_size: размер блока для DWT (должен быть равен 16).
        :param delta: шаг квантования для QIM.
        :param wavelet_name: тип вейвлета (haar, db4, sym4).
        :return output_image: матрица изображения с встроенным ЦВЗ.
        """
        defaults = {
            "input_image": None,
            "watermark_bits": None,
            "block_size": 16,
            "delta": 40.0,
            "wavelet_name": "haar"
        }
        kwargs = {**defaults, **args}
        image = kwargs.get("input_image")
        if image is None:
            image = kwargs.get("image_path")
        watermark = kwargs.get("watermark_bits")
        if image is None or watermark is None:
            raise ValueError("Не переданы input_image/image_path или watermark_bits")

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(image, dtype=np.float64)
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] wm_c = np.ascontiguousarray(watermark, dtype=np.int32)
        cdef int block_size = kwargs["block_size"]
        cdef double delta = kwargs["delta"]
        cdef bytes wavelet_name = kwargs["wavelet_name"].encode('utf-8')

        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]

        if block_size != 16:
            raise ValueError("block_size должен быть равен 16")

        cdef int blocks_h = H // block_size
        cdef int blocks_w = W // block_size
        cdef int capacity = blocks_h * blocks_w
        cdef int wm_len = wm_c.shape[0]

        if wm_len > capacity:
            raise ValueError(
                f"Не хватает ёмкости: нужно {wm_len} блоков, доступно {capacity}. "
                f"Уменьшите длину ЦВЗ или увеличьте размер изображения."
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

        _embed_core(img_view, blocks_h, blocks_w, wm_c, block_size, delta,
                   block_np, LL_np, LH_np, HL_np, HH_np,
                   &temp_np[0], &row_a_np[0], &row_d_np[0],
                   &col_in_np[0], &col_a_np[0], &col_d_np[0],
                   &col_out_np[0], &row_out_np[0],
                   h_buf, g_buf, L_buf)

        return watermarked_img

    @staticmethod
    def extraction(**args):
        """
        Извлекает биты ЦВЗ из сингулярных чисел низкочастотной подматрицы
        DCT-коэффициентов LL-подполосы после DWT.

        :param input_image: матрица изображения с ЦВЗ (канал яркости Y).
        :param num_bits: длина ЦВЗ.
        :param block_size: размер блока для DWT.
        :param delta: шаг квантования.
        :param wavelet_name: тип вейвлета.
        :return extracted_wm: извлечённый ЦВЗ.
        """
        defaults = {
            "input_image": None,
            "num_bits": 0,
            "block_size": 16,
            "delta": 40.0,
            "wavelet_name": "haar"
        }
        kwargs = {**defaults, **args}
        image = kwargs.get("input_image")
        if image is None:
            image = kwargs.get("image_path")
        num_bits = kwargs.get("num_bits")
        if image is None or not num_bits:
            raise ValueError("Не переданы input_image/image_path или num_bits")

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(image, dtype=np.float64)
        cdef int wm_length = num_bits
        cdef int block_size = kwargs["block_size"]
        cdef double delta = kwargs["delta"]
        cdef bytes wavelet_name = kwargs["wavelet_name"].encode('utf-8')

        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]

        if block_size != 16:
            raise ValueError("block_size должен быть равен 16")

        cdef int blocks_h = H // block_size
        cdef int blocks_w = W // block_size
        cdef int capacity = blocks_h * blocks_w

        if wm_length > capacity:
            raise ValueError(f"wm_length превышает число блоков.")

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

        _extract_core(img_view, blocks_h, blocks_w, extracted, wm_length, block_size, delta,
                    block_np, LL_np, LH_np, HL_np, HH_np,
                    &temp_np[0], &row_a_np[0], &row_d_np[0],
                    &col_in_np[0], &col_a_np[0], &col_d_np[0],
                    &col_out_np[0], &row_out_np[0],
                    h_buf, g_buf, L_buf)

        return extracted_wm