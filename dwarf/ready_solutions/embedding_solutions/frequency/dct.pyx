"""
Метод DCT (Discrete Cosine Transform) — встраивание в DCT-коэффициенты блоков 8×8.
https://scispace.com/pdf/towards-robust-and-hidden-image-copyright-labeling-1hriyt4461.pdf
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport init_dct_matrix, apply_dct_8x8, apply_idct_8x8, DBlock

cnp.import_array()

cdef void _embed_core(double[:, :] img_view, int blocks_h, int blocks_w,
                      int[:] watermark, double margin, double threshold,
                      int c1_row, int c1_col, int c2_row, int c2_col):
    """
    Встраивание ЦВЗ.
    
    Args:
        img_view: представление изображения для записи.
        blocks_h: количество блоков по высоте.
        blocks_w: количество блоков по ширине.
        watermark: ЦВЗ.
        margin: величина, на которую модуль модифицируемого коэффициента
            делается больше модуля второго коэффициента пары при встраивании.
        threshold: минимально допустимая по модулю разность коэффициентов
            пары, при которой блок уже пригоден для передачи нужного бита и не
            требует модификации.
        c1_row, c1_col: индекс первого DCT-коэффициента пары.
        c2_row, c2_col: индекс второго DCT-коэффициента пары.
    """
    cdef DBlock block_img, block_dct, block_idct_arr
    cdef int b_idx = 0
    cdef int wm_len = watermark.shape[0]
    cdef int bi, bj, r, c
    cdef double c1, c2, k

    for bi in range(blocks_h):
        for bj in range(blocks_w):
            if b_idx >= wm_len:
                break
            for r in range(8):
                for c in range(8):
                    block_img[r][c] = img_view[bi*8 + r, bj*8 + c]

            apply_dct_8x8(block_img, block_dct)

            c1 = block_dct[c1_row][c1_col]
            c2 = block_dct[c2_row][c2_col]
            k = fabs(c1) - fabs(c2)

            if watermark[b_idx] == 1:
                if k <= threshold:
                    block_dct[c1_row][c1_col] = fabs(c2) + margin if c1 >= 0 else -(fabs(c2) + margin)
            else:
                if k >= -threshold:
                    block_dct[c2_row][c2_col] = fabs(c1) + margin if c2 >= 0 else -(fabs(c1) + margin)

            apply_idct_8x8(block_dct, block_idct_arr)

            for r in range(8):
                for c in range(8):
                    img_view[bi*8 + r, bj*8 + c] = block_idct_arr[r][c]
            b_idx += 1
        if b_idx >= wm_len:
            break

cdef void _extract_core(double[:, :] img_view, int blocks_h, int blocks_w,
                        int[:] extracted, int wm_length, double threshold,
                        int c1_row, int c1_col, int c2_row, int c2_col):
    """
    Извлечение ЦВЗ.
    
    Args:
        img_view: представление изображения с ЦВЗ.
        blocks_h: количество блоков по высоте.
        blocks_w: количество блоков по ширине.
        extracted: выходной массив для извлечённых бит.
        wm_length: длина ЦВЗ.
        threshold: минимально допустимая по модулю разность коэффициентов
            пары, при которой бит считается надёжно определённым.
            Должно совпадать со значением в embed_watermark_dct.
        c1_row, c1_col: индекс первого DCT-коэффициента пары.
        c2_row, c2_col: индекс второго DCT-коэффициента пары.
    """
    cdef DBlock block_img, block_dct
    cdef int b_idx = 0
    cdef int bi, bj, r, c
    cdef double c1, c2, k

    for bi in range(blocks_h):
        for bj in range(blocks_w):
            if b_idx >= wm_length:
                break
            for r in range(8):
                for c in range(8):
                    block_img[r][c] = img_view[bi*8 + r, bj*8 + c]
            
            apply_dct_8x8(block_img, block_dct)
            
            c1 = block_dct[c1_row][c1_col]
            c2 = block_dct[c2_row][c2_col]
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


class DCT(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        """
        Встраивает биты ЦВЗ в DCT-коэффициенты блоков 8x8 изображения.
        :param input_image: матрица входного изображения.
        :param watermark_bits: массив битов ЦВЗ.
        :param margin: величина, на которую модуль модифицируемого коэффициента
            делается больше модуля второго коэффициента пары при встраивании.
        :param threshold: минимально допустимая по модулю разность коэффициентов
            пары, при которой блок уже пригоден для передачи нужного бита и не
            требует модификации.
        :param coef1_row: строка первого DCT-коэффициента пары (по умолчанию 3).
        :param coef1_col: столбец первого DCT-коэффициента пары (по умолчанию 4).
        :param coef2_row: строка второго DCT-коэффициента пары (по умолчанию 4).
        :param coef2_col: столбец второго DCT-коэффициента пары (по умолчанию 3).
            Пара по умолчанию сохранена для обратной совместимости. Она
            чувствительна к Gaussian blur около sigma=1; для исследований
            можно выбрать более низкочастотную пару без изменения ядра.

        :return output_image: матрица изображения с встроенным ЦВЗ.
        """
        defaults = {
                    "input_image": None,
                    "watermark_bits": None,
                    "margin": 150.0,
                    "threshold": 25.0,
                    "coef1_row": 3,
                    "coef1_col": 4,
                    "coef2_row": 4,
                    "coef2_col": 3
                }
        args = {**defaults, **args}
        
        image = args["input_image"]
        watermark = args["watermark_bits"]
        
        if image is None or watermark is None:
            raise ValueError("input_image/image_path or watermark_bits not given")

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(image, dtype=np.float64)
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] wm_c = np.ascontiguousarray(watermark, dtype=np.int32)
        cdef double margin = args["margin"]
        cdef double threshold = args["threshold"]
        cdef int c1_row = int(args["coef1_row"])
        cdef int c1_col = int(args["coef1_col"])
        cdef int c2_row = int(args["coef2_row"])
        cdef int c2_col = int(args["coef2_col"])

        if not (0 <= c1_row < 8 and 0 <= c1_col < 8 and
                0 <= c2_row < 8 and 0 <= c2_col < 8):
            raise ValueError("DCT coefficient indices must be in range [0, 7].")
        if c1_row == c2_row and c1_col == c2_col:
            raise ValueError("DCT coefficient pair must contain two distinct coefficients.")
        if (c1_row == 0 and c1_col == 0) or (c2_row == 0 and c2_col == 0):
            raise ValueError("The DC coefficient (0, 0) cannot be used for watermark embedding.")
        
        init_dct_matrix()
        
        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]
        cdef int blocks_h = H // 8
        cdef int blocks_w = W // 8
        cdef int capacity = blocks_h * blocks_w
        cdef int wm_len = wm_c.shape[0]

        if wm_len > capacity:
            raise ValueError(
                f"Not enough capacity: need {wm_len} blocks, available {capacity}."
            )
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] watermarked_img = img_c.copy()
        cdef double[:, :] img_view = watermarked_img
        
        _embed_core(
            img_view, blocks_h, blocks_w, wm_c, margin, threshold,
            c1_row, c1_col, c2_row, c2_col
        )
        
        return watermarked_img

    @staticmethod
    def extraction(**args):
        """
        Извлекает биты ЦВЗ из DCT-коэффициентов блоков 8x8 изображения.
        :param input_image: матрица изображения с ЦВЗ.
        :param num_bits: длина ЦВЗ.
        :param threshold: минимально допустимая по модулю разность коэффициентов
            пары, при которой бит считается надёжно определённым.
            Должно совпадать со значением при встраивании.
        :param coef1_row: строка первого DCT-коэффициента пары.
        :param coef1_col: столбец первого DCT-коэффициента пары.
        :param coef2_row: строка второго DCT-коэффициента пары.
        :param coef2_col: столбец второго DCT-коэффициента пары.
            Все четыре индекса должны совпадать со значениями при встраивании.

        :return extracted_wm: извлечённый ЦВЗ.
            Значение -1 в элементе массива означает, что бит не удалось надёжно определить.
        """
        defaults = {
                    "input_image": None,
                    "num_bits": 0,
                    "threshold": 25.0,
                    "coef1_row": 3,
                    "coef1_col": 4,
                    "coef2_row": 4,
                    "coef2_col": 3
                }
        args = {**defaults, **args}
        
        image = args["input_image"]
        num_bits = args["num_bits"]
        
        if image is None or not num_bits:
            raise ValueError("input_image/image_path or num_bits not given")

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(image, dtype=np.float64)
        cdef double threshold = args["threshold"]
        cdef int c1_row = int(args["coef1_row"])
        cdef int c1_col = int(args["coef1_col"])
        cdef int c2_row = int(args["coef2_row"])
        cdef int c2_col = int(args["coef2_col"])

        if not (0 <= c1_row < 8 and 0 <= c1_col < 8 and
                0 <= c2_row < 8 and 0 <= c2_col < 8):
            raise ValueError("DCT coefficient indices must be in range [0, 7].")
        if c1_row == c2_row and c1_col == c2_col:
            raise ValueError("DCT coefficient pair must contain two distinct coefficients.")
        if (c1_row == 0 and c1_col == 0) or (c2_row == 0 and c2_col == 0):
            raise ValueError("The DC coefficient (0, 0) cannot be used for watermark extraction.")
        
        init_dct_matrix()
        
        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]
        cdef int blocks_h = H // 8
        cdef int blocks_w = W // 8
        cdef int capacity = blocks_h * blocks_w

        if num_bits > capacity:
            raise ValueError(
                f"Cannot extract {num_bits} bits: capacity is {capacity}."
            )
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] extracted_wm = np.zeros(num_bits, dtype=np.int32)
        cdef double[:, :] img_view = img_c
        
        _extract_core(
            img_view, blocks_h, blocks_w, extracted_wm, num_bits, threshold,
            c1_row, c1_col, c2_row, c2_col
        )
        
        return extracted_wm