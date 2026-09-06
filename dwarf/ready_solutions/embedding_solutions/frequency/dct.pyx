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
                        int[:] extracted, int wm_length,
                        int c1_row, int c1_col, int c2_row, int c2_col):
    """
    Извлечение ЦВЗ.
    
    Args:
        img_view: представление изображения с ЦВЗ.
        blocks_h: количество блоков по высоте.
        blocks_w: количество блоков по ширине.
        extracted: выходной массив для извлечённых бит.
        wm_length: длина ЦВЗ.
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

            extracted[b_idx] = 1 if k >= 0.0 else 0
            b_idx += 1
        if b_idx >= wm_length:
            break


class DCT(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        """
        Встраивает биты ЦВЗ в DCT-коэффициенты блоков 8x8 изображения.
        :param input_image: RGB-изображение uint8 формы (H, W, 3).
        :param watermark_bits: массив uint8 из значений 0/1.
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

        :return output_image: RGB-изображение uint8 формы (H, W, 3) со встроенным ЦВЗ.
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
        
        watermarked_y = np.asarray(watermarked_img, dtype=np.float64)
        delta_y = watermarked_y - input_y
        output_rgb = rgb_c.astype(np.float64) + delta_y[:, :, None]
        return np.ascontiguousarray(
            np.clip(np.rint(output_rgb), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def extraction(**args):
        """
        Извлекает биты ЦВЗ из DCT-коэффициентов блоков 8x8 изображения.
        :param input_image: RGB-изображение uint8 формы (H, W, 3) с ЦВЗ.
        :param num_bits: длина ЦВЗ.
        :param coef1_row: строка первого DCT-коэффициента пары.
        :param coef1_col: столбец первого DCT-коэффициента пары.
        :param coef2_row: строка второго DCT-коэффициента пары.
        :param coef2_col: столбец второго DCT-коэффициента пары.
            Все четыре индекса должны совпадать со значениями при встраивании.

        :return extracted_wm: извлечённый int8-массив из значений 0/1.
        """
        defaults = {
                    "input_image": None,
                    "num_bits": 0,
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
            img_view, blocks_h, blocks_w, extracted_wm, num_bits,
            c1_row, c1_col, c2_row, c2_col
        )
        
        if np.any((extracted_wm != 0) & (extracted_wm != 1)):
            raise RuntimeError("dct extraction produced a non-binary watermark")
        return np.ascontiguousarray(extracted_wm, dtype=np.int8)