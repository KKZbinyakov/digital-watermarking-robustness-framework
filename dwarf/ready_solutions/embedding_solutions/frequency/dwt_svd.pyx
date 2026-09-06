"""
Гибридный метод DWT-SVD — встраивание в сингулярные числа LL-подполосы
после дискретного вейвлет-преобразования блоков.
"""

import numpy as np
cimport numpy as cnp

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    get_wavelet_filters, dwt_2d_block, idwt_2d_block,
    power_iteration, power_iteration_sigma,
    qim_embed, qim_extract,
    N_SVD
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
    cdef int b_idx = 0
    cdef int wm_len = watermark.shape[0]
    cdef int bi, bj, i, j, r, c

    cdef double u[N_SVD]
    cdef double v[N_SVD]
    cdef double s1, s_new, delta_sigma

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

            power_iteration(&LL[0, 0], u, v, &s1)

            if s1 < 1e-12:
                b_idx += 1
                continue

            s_new = qim_embed(s1, watermark[b_idx], delta)
            delta_sigma = s_new - s1

            for i in range(N_SVD):
                for j in range(N_SVD):
                    LL[i, j] += delta_sigma * u[i] * v[j]

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
    cdef int b_idx = 0
    cdef int bi, bj, r, c
    cdef double s1

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

            s1 = power_iteration_sigma(&LL[0, 0])
            extracted[b_idx] = qim_extract(s1, delta)

            b_idx += 1
        if b_idx >= wm_length:
            break

class DWTSVD(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        """
        Встраивает биты ЦВЗ в сингулярные числа LL-подполосы после DWT.

        :param input_image: RGB-изображение uint8 формы (H, W, 3).
        :param watermark_bits: массив uint8 из значений 0/1.
        :param block_size: размер блока для DWT (должен быть равен 2 * N_SVD).
        :param delta: шаг квантования для QIM.
        :param wavelet_name: тип вейвлета (haar, db4, sym4).
        :return output_image: RGB-изображение uint8 формы (H, W, 3) со встроенным ЦВЗ.
        """
        defaults = {
                    "input_image": None,
                    "watermark_bits": None,
                    "block_size": 8,
                    "delta": 40.0,
                    "wavelet_name": "haar"
                }
        args = {**defaults, **args}
        image = args["input_image"]
        watermark = args["watermark_bits"]
        if image is None or watermark is None:
            raise ValueError("input_image/image_path or watermark_bits not given")

        image_arr = np.asarray(image)
        if image_arr.ndim != 3 or image_arr.shape[2] != 3:
            raise ValueError(
                f"input_image must have shape (H, W, 3), got {image_arr.shape}"
            )
        if image_arr.dtype != np.uint8:
            raise TypeError(
                f"input_image must have dtype uint8, got {image_arr.dtype}"
            )

        watermark_arr = np.asarray(watermark)
        if watermark_arr.ndim != 1:
            raise ValueError(
                f"watermark_bits must be one-dimensional, got shape {watermark_arr.shape}"
            )
        if watermark_arr.dtype != np.uint8:
            raise TypeError(
                f"watermark_bits must have dtype uint8, got {watermark_arr.dtype}"
            )
        if watermark_arr.size == 0:
            raise ValueError("watermark_bits must not be empty")
        if np.any((watermark_arr != 0) & (watermark_arr != 1)):
            raise ValueError("watermark_bits must contain only 0 and 1")

        cdef cnp.ndarray[cnp.uint8_t, ndim=3, mode='c'] rgb_c = np.ascontiguousarray(image_arr)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] input_y = np.ascontiguousarray(
            0.299 * rgb_c[:, :, 0]
            + 0.587 * rgb_c[:, :, 1]
            + 0.114 * rgb_c[:, :, 2],
            dtype=np.float64,
        )
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = input_y.copy()
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] wm_c = np.ascontiguousarray(
            watermark_arr, dtype=np.int32
        )
        cdef int block_size = int(args["block_size"])
        cdef double delta = args["delta"]
        cdef bytes wavelet_name = args["wavelet_name"].encode('utf-8')

        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]

        if block_size != 2 * N_SVD:
            raise ValueError(f"block_size must be {2 * N_SVD}")

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

        watermarked_y = np.asarray(watermarked_img, dtype=np.float64)
        delta_y = watermarked_y - input_y
        output_rgb = rgb_c.astype(np.float64) + delta_y[:, :, None]
        return np.ascontiguousarray(
            np.clip(np.rint(output_rgb), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def extraction(**args):
        """
        Извлекает биты ЦВЗ из сингулярных чисел LL-подполосы после DWT.

        :param input_image: RGB-изображение uint8 формы (H, W, 3) с ЦВЗ.
        :param num_bits: длина ЦВЗ.
        :param block_size: размер блока для DWT.
        :param delta: шаг квантования.
        :param wavelet_name: тип вейвлета.
        :return extracted_wm: извлечённый int8-массив из значений 0/1.
        """
        defaults = {
                    "input_image": None,
                    "num_bits": 0,
                    "block_size": 8,
                    "delta": 40.0,
                    "wavelet_name": "haar"
                }
        args = {**defaults, **args}
        image = args["input_image"]
        num_bits = args["num_bits"]
        if image is None or not num_bits:
            raise ValueError("input_image/image_path or watermark_bits not given")

        image_arr = np.asarray(image)
        if image_arr.ndim != 3 or image_arr.shape[2] != 3:
            raise ValueError(
                f"input_image must have shape (H, W, 3), got {image_arr.shape}"
            )
        if image_arr.dtype != np.uint8:
            raise TypeError(
                f"input_image must have dtype uint8, got {image_arr.dtype}"
            )

        cdef cnp.ndarray[cnp.uint8_t, ndim=3, mode='c'] rgb_c = np.ascontiguousarray(image_arr)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(
            0.299 * rgb_c[:, :, 0]
            + 0.587 * rgb_c[:, :, 1]
            + 0.114 * rgb_c[:, :, 2],
            dtype=np.float64,
        )
        cdef int wm_length = num_bits
        cdef int block_size = int(args["block_size"])
        cdef double delta = args["delta"]
        cdef bytes wavelet_name = args["wavelet_name"].encode('utf-8')

        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]

        if block_size != 2 * N_SVD:
            raise ValueError(f"block_size must be {2 * N_SVD}")

        cdef int blocks_h = H // block_size
        cdef int blocks_w = W // block_size
        cdef int capacity = blocks_h * blocks_w

        if wm_length > capacity:
            raise ValueError(f"wm_length is more than the number of blocks.")

        cdef double h_buf[32]
        cdef double g_buf[32]
        cdef int L_buf
        get_wavelet_filters(wavelet_name, h_buf, g_buf, &L_buf)

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

        if np.any((extracted_wm != 0) & (extracted_wm != 1)):
            raise RuntimeError("dwt_svd extraction produced a non-binary watermark")
        return np.ascontiguousarray(extracted_wm, dtype=np.int8)