"""
Метод DCT (Discrete Cosine Transform) — встраивание в DCT-коэффициенты блоков 8×8.
https://scispace.com/pdf/towards-robust-and-hidden-image-copyright-labeling-1hriyt4461.pdf
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils cimport init_dct_matrix, apply_dct_8x8, apply_idct_8x8, DBlock

cnp.import_array()

cdef void _embed_core(double[:, :] img_view, int blocks_h, int blocks_w, 
                      int[:] watermark, double margin, double threshold):
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

            c1 = block_dct[3][4]
            c2 = block_dct[4][3]
            k = fabs(c1) - fabs(c2)

            if watermark[b_idx] == 1:
                if k <= threshold:
                    block_dct[3][4] = fabs(c2) + margin if c1 >= 0 else -(fabs(c2) + margin)
            else:
                if k >= -threshold:
                    block_dct[4][3] = fabs(c1) + margin if c2 >= 0 else -(fabs(c1) + margin)

            apply_idct_8x8(block_dct, block_idct_arr)

            for r in range(8):
                for c in range(8):
                    img_view[bi*8 + r, bj*8 + c] = block_idct_arr[r][c]
            b_idx += 1
        if b_idx >= wm_len:
            break

cdef void _extract_core(double[:, :] img_view, int blocks_h, int blocks_w, 
                        int[:] extracted, int wm_length, double threshold):
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
            
            c1 = block_dct[3][4]
            c2 = block_dct[4][3]
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

        :return output_image: матрица изображения с встроенным ЦВЗ.
        """
        defaults = {"input_image": None, "watermark_bits": None, "margin": 150.0, "threshold": 25.0}
        kwargs = {**defaults, **args}
        
        image = kwargs.get("input_image")
        if image is None:
            image = kwargs.get("image_path")
        watermark = kwargs.get("watermark_bits")
        
        if image is None or watermark is None:
            raise ValueError("Не переданы input_image/image_path или watermark_bits")

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(image, dtype=np.float64)
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] wm_c = np.ascontiguousarray(watermark, dtype=np.int32)
        cdef double margin = kwargs["margin"]
        cdef double threshold = kwargs["threshold"]
        
        init_dct_matrix()
        
        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] watermarked_img = img_c.copy()
        cdef double[:, :] img_view = watermarked_img
        
        _embed_core(img_view, H // 8, W // 8, wm_c, margin, threshold)
        
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

        :return extracted_wm: извлечённый ЦВЗ.
            Значение -1 в элементе массива означает, что бит не удалось надёжно определить.
        """
        defaults = {"input_image": None, "num_bits": 0, "threshold": 25.0}
        kwargs = {**defaults, **args}
        
        image = kwargs.get("input_image")
        if image is None:
            image = kwargs.get("image_path")
        num_bits = kwargs.get("num_bits")
        
        if image is None or not num_bits:
            raise ValueError("Не переданы input_image/image_path или num_bits")

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(image, dtype=np.float64)
        cdef double threshold = kwargs["threshold"]
        
        init_dct_matrix()
        
        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] extracted_wm = np.zeros(num_bits, dtype=np.int32)
        cdef double[:, :] img_view = img_c
        
        _extract_core(img_view, H // 8, W // 8, extracted_wm, num_bits, threshold)
        
        return extracted_wm