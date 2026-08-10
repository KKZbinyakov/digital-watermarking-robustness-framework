"""
Гибридный метод DCT-SVD — встраивание в сингулярные числа низкочастотной
подматрицы DCT-коэффициентов блоков 8x8.
"""

import numpy as np
cimport numpy as cnp

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pxd cimport (
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

        :param input_image: матрица входного изображения (канал яркости Y).
        :param watermark_bits: массив битов ЦВЗ.
        :param delta: шаг квантования для QIM.
        :return output_image: матрица изображения с встроенным ЦВЗ.
        """
        defaults = {
            "input_image": None,
            "watermark_bits": None,
            "delta": 40.0
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
        cdef double delta = kwargs["delta"]

        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]

        cdef int blocks_h = H // 8
        cdef int blocks_w = W // 8
        cdef int capacity = blocks_h * blocks_w
        cdef int wm_len = wm_c.shape[0]

        if wm_len > capacity:
            raise ValueError(
                f"Не хватает ёмкости: нужно {wm_len} блоков, доступно {capacity}. "
                f"Уменьшите длину ЦВЗ или увеличьте размер изображения."
            )

        init_dct_matrix()

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] watermarked_img = img_c.copy()
        cdef double[:, :] img_view = watermarked_img

        _embed_core(img_view, blocks_h, blocks_w, wm_c, delta)

        return watermarked_img

    @staticmethod
    def extraction(**args):
        """
        Извлекает биты ЦВЗ из сингулярных чисел низкочастотной подматрицы
        DCT-коэффициентов блоков 8x8.

        :param input_image: матрица изображения с ЦВЗ (канал яркости Y).
        :param num_bits: длина ЦВЗ.
        :param delta: шаг квантования.
        :return extracted_wm: извлечённый ЦВЗ.
        """
        defaults = {
            "input_image": None,
            "num_bits": 0,
            "delta": 40.0
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
        cdef double delta = kwargs["delta"]

        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]

        cdef int blocks_h = H // 8
        cdef int blocks_w = W // 8
        cdef int capacity = blocks_h * blocks_w

        if wm_length > capacity:
            raise ValueError(f"wm_length превышает число блоков 8x8.")

        init_dct_matrix()

        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] extracted_wm = np.zeros(wm_length, dtype=np.int32)
        cdef int[:] extracted = extracted_wm

        cdef double[:, :] img_view = img_c

        _extract_core(img_view, blocks_h, blocks_w, extracted, wm_length, delta)

        return extracted_wm