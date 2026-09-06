"""
Гибридный метод DCT-SVD — встраивание в сингулярные числа низкочастотной
подматрицы DCT-коэффициентов блоков 8x8.
"""

import numpy as np
cimport numpy as cnp

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    init_dct_matrix, apply_dct_8x8, apply_idct_8x8, DBlock,
    power_iteration, power_iteration_sigma,
    qim_embed, qim_extract,
    N_SVD, NN_SVD
)

cnp.import_array()

cdef void _embed_core(double[:, :] img_view, int blocks_h, int blocks_w,
                      int[:] watermark, double delta):
    """
    Ядро встраивания ЦВЗ.

    Args:
        img_view: представление изображения для записи.
        blocks_h: количество блоков по высоте.
        blocks_w: количество блоков по ширине.
        watermark: ЦВЗ.
        delta: шаг квантования.
    """
    cdef DBlock block_img, block_dct, block_idct_arr
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

            for r in range(8):
                for c in range(8):
                    block_img[r][c] = img_view[bi * 8 + r, bj * 8 + c]

            apply_dct_8x8(block_img, block_dct)

            for i in range(N_SVD):
                for j in range(N_SVD):
                    svd_matrix[i * N_SVD + j] = block_dct[i][j]

            power_iteration(svd_matrix, u, v, &s1)

            if s1 < 1e-12:
                b_idx += 1
                continue

            s_new = qim_embed(s1, watermark[b_idx], delta)
            delta_sigma = s_new - s1

            for i in range(N_SVD):
                for j in range(N_SVD):
                    block_dct[i][j] += delta_sigma * u[i] * v[j]

            apply_idct_8x8(block_dct, block_idct_arr)

            for r in range(8):
                for c in range(8):
                    img_view[bi * 8 + r, bj * 8 + c] = block_idct_arr[r][c]

            b_idx += 1
        if b_idx >= wm_len:
            break

cdef void _extract_core(double[:, :] img_view, int blocks_h, int blocks_w,
                        int[:] extracted, int wm_length, double delta):
    """
    Ядро извлечения ЦВЗ.

    Args:
        img_view: представление изображения с ЦВЗ.
        blocks_h: количество блоков по высоте.
        blocks_w: количество блоков по ширине.
        extracted: выходной массив для извлечённых бит.
        wm_length: длина ЦВЗ.
        delta: шаг квантования.
    """
    cdef DBlock block_img, block_dct
    cdef double svd_matrix[NN_SVD]
    cdef double s1

    cdef int b_idx = 0
    cdef int bi, bj, i, j, r, c

    for bi in range(blocks_h):
        for bj in range(blocks_w):
            if b_idx >= wm_length:
                break

            for r in range(8):
                for c in range(8):
                    block_img[r][c] = img_view[bi * 8 + r, bj * 8 + c]

            apply_dct_8x8(block_img, block_dct)

            for i in range(N_SVD):
                for j in range(N_SVD):
                    svd_matrix[i * N_SVD + j] = block_dct[i][j]

            s1 = power_iteration_sigma(svd_matrix)
            extracted[b_idx] = qim_extract(s1, delta)

            b_idx += 1
        if b_idx >= wm_length:
            break

class DCT_SVD(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        """
        Встраивает биты ЦВЗ в сингулярные числа низкочастотной подматрицы
        DCT-коэффициентов блоков 8x8.

        :param input_image: RGB-изображение uint8 формы (H, W, 3).
        :param watermark_bits: массив uint8 из значений 0/1.
        :param delta: шаг квантования для QIM.
        :return output_image: RGB-изображение uint8 формы (H, W, 3) со встроенным ЦВЗ.
        """
        defaults = {
                    "input_image": None,
                    "watermark_bits": None,
                    "delta": 40.0
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
        cdef double delta = args["delta"]

        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]

        cdef int blocks_h = H // 8
        cdef int blocks_w = W // 8
        cdef int capacity = blocks_h * blocks_w
        cdef int wm_len = wm_c.shape[0]

        if wm_len > capacity:
            raise ValueError(
                f"Not enough capacity: neede {wm_len} blocks, available {capacity}."
            )

        init_dct_matrix()

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] watermarked_img = img_c.copy()
        cdef double[:, :] img_view = watermarked_img

        _embed_core(img_view, blocks_h, blocks_w, wm_c, delta)

        watermarked_y = np.asarray(watermarked_img, dtype=np.float64)
        delta_y = watermarked_y - input_y
        output_rgb = rgb_c.astype(np.float64) + delta_y[:, :, None]
        return np.ascontiguousarray(
            np.clip(np.rint(output_rgb), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def extraction(**args):
        """
        Извлекает биты ЦВЗ из сингулярных чисел низкочастотной подматрицы
        DCT-коэффициентов блоков 8x8.

        :param input_image: RGB-изображение uint8 формы (H, W, 3) с ЦВЗ.
        :param num_bits: длина ЦВЗ.
        :param delta: шаг квантования.
        :return extracted_wm: извлечённый int8-массив из значений 0/1.
        """
        defaults = {
                    "input_image": None,
                    "num_bits": 0,
                    "delta": 40.0
                }
        args = {**defaults, **args}
        image = args["input_image"]
        num_bits = args["num_bits"]
        if image is None or not num_bits:
            raise ValueError("input_image/image_path or num_bits not given")

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
        cdef double delta = args["delta"]

        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]

        cdef int blocks_h = H // 8
        cdef int blocks_w = W // 8
        cdef int capacity = blocks_h * blocks_w

        if wm_length > capacity:
            raise ValueError(f"wm_length is more than the number of blocks 8x8.")

        init_dct_matrix()

        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] extracted_wm = np.zeros(wm_length, dtype=np.int32)
        cdef int[:] extracted = extracted_wm

        cdef double[:, :] img_view = img_c

        _extract_core(img_view, blocks_h, blocks_w, extracted, wm_length, delta)

        if np.any((extracted_wm != 0) & (extracted_wm != 1)):
            raise RuntimeError("dct_svd extraction produced a non-binary watermark")
        return np.ascontiguousarray(extracted_wm, dtype=np.int8)