"""
Метод DWT (Discrete Wavelet Transform) — в подполосы LL, LH, HL, HH (Haar, Daubechies, Symlets).
https://ictactjournals.in/paper/IJIVP_V6_I2_paper_5_1133_1136.pdf
"""

import numpy as np
cimport numpy as cnp

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils cimport (
    get_wavelet_filters, dwt_2d_block, idwt_2d_block
)

cnp.import_array()

cdef void _embed_core(double[:, :] img_view, int blocks_h, int blocks_w, 
                      int[:] watermark, int block_size, double min_difference,
                      double amplification_factor, bint redundant,
                      double[:, :] LL, double[:, :] LH, double[:, :] HL, double[:, :] HH,
                      double[:, :] block, double* temp_ptr, double* row_a_ptr,
                      double* row_d_ptr, double* col_in_ptr, double* col_a_ptr,
                      double* col_d_ptr, double* col_out_ptr, double* row_out_ptr,
                      const double* h, const double* g, int L):
    """
    Встраивание ЦВЗ.
    
    Args:
        img_view: представление изображения для записи.
        blocks_h: количество блоков по высоте.
        blocks_w: количество блоков по ширине.
        watermark: ЦВЗ.
        block_size: размер блока.
        min_difference: минимальная разность между коэффициентами.
        amplification_factor: коэффициент усиления.
        redundant: использовать избыточное встраивание.
        LL, LH, HL, HH: буферы подполос.
        block: буфер блока.
        temp_ptr, row_a_ptr, row_d_ptr, col_in_ptr, col_a_ptr, col_d_ptr, col_out_ptr, row_out_ptr: указатели на буферы.
        h, g: фильтры.
        L: длина фильтра.
    """
    cdef int wm_len = watermark.shape[0]
    cdef int copies = 3 if redundant else 1
    cdef int bi, bj, r, c, copy_idx, b_idx
    cdef double hl_val, lh_val, diff
    cdef double[:, :] hl_view = HL
    cdef double[:, :] lh_view = LH
    cdef double[:, :] block_view = block
    
    for copy_idx in range(copies):
        b_idx = 0
        for bi in range(blocks_h):
            for bj in range(blocks_w):
                if b_idx >= wm_len:
                    break
                
                for r in range(block_size):
                    for c in range(block_size):
                        block_view[r, c] = img_view[bi * block_size + r, bj * block_size + c]
                
                dwt_2d_block(block, LL, LH, HL, HH, temp_ptr, row_a_ptr, row_d_ptr, 
                            col_in_ptr, col_a_ptr, col_d_ptr, h, g, L, block_size)
                
                hl_val = hl_view[0, 0]
                lh_val = lh_view[0, 0]
                
                if watermark[b_idx] == 1:
                    if hl_val <= lh_val:
                        hl_view[0, 0] = lh_val * amplification_factor
                    
                    diff = hl_view[0, 0] - lh_view[0, 0]
                    if diff < min_difference:
                        hl_view[0, 0] = lh_view[0, 0] + min_difference
                
                else:
                    if lh_val <= hl_val:
                        lh_view[0, 0] = hl_val * amplification_factor
                    
                    diff = lh_view[0, 0] - hl_view[0, 0]
                    if diff < min_difference:
                        lh_view[0, 0] = hl_view[0, 0] + min_difference
                
                idwt_2d_block(LL, LH, HL, HH, block, temp_ptr, col_a_ptr, col_d_ptr, 
                             col_out_ptr, row_a_ptr, row_d_ptr, row_out_ptr, h, g, L, block_size)
                
                for r in range(block_size):
                    for c in range(block_size):
                        img_view[bi * block_size + r, bj * block_size + c] = block_view[r, c]
                
                b_idx += 1
            if b_idx >= wm_len:
                break
        if not redundant:
            break

cdef void _extract_core(double[:, :] img_view, int blocks_h, int blocks_w, 
                        int[:] extracted, int wm_length, int block_size, bint redundant,
                        double[:, :] LL, double[:, :] LH, double[:, :] HL, double[:, :] HH,
                        double[:, :] block, double* temp_ptr, double* row_a_ptr,
                        double* row_d_ptr, double* col_in_ptr, double* col_a_ptr,
                        double* col_d_ptr, const double* h, const double* g, int L):
    """
    Извлечение ЦВЗ.
    
    Args:
        img_view: представление изображения с ЦВЗ.
        blocks_h: количество блоков по высоте.
        blocks_w: количество блоков по ширине.
        extracted: выходной массив для извлечённых бит.
        wm_length: длина ЦВЗ.
        block_size: размер блока.
        redundant: использовать избыточное извлечение.
        LL, LH, HL, HH: буферы подполос.
        block: буфер блока.
        temp_ptr, row_a_ptr, row_d_ptr, col_in_ptr, col_a_ptr, col_d_ptr: указатели на буферы.
        h, g: фильтры.
        L: длина фильтра.
    """
    cdef int bi, bj, r, c, copy_idx, b_idx
    cdef double hl_val, lh_val
    cdef int votes_1
    cdef double[:, :] hl_view = HL
    cdef double[:, :] lh_view = LH
    cdef double[:, :] block_view = block
    
    cdef cnp.ndarray[cnp.int32_t, ndim=2, mode='c'] votes = np.zeros((3, wm_length), dtype=np.int32)
    
    if redundant:
        for copy_idx in range(3):
            b_idx = 0
            for bi in range(blocks_h):
                for bj in range(blocks_w):
                    if b_idx >= wm_length:
                        break
                    
                    for r in range(block_size):
                        for c in range(block_size):
                            block_view[r, c] = img_view[bi * block_size + r, bj * block_size + c]
                    
                    dwt_2d_block(block, LL, LH, HL, HH, temp_ptr, row_a_ptr, row_d_ptr, 
                                col_in_ptr, col_a_ptr, col_d_ptr, h, g, L, block_size)
                    
                    hl_val = hl_view[0, 0]
                    lh_val = lh_view[0, 0]
                    
                    if hl_val > lh_val:
                        votes[copy_idx, b_idx] = 1
                    else:
                        votes[copy_idx, b_idx] = 0
                    
                    b_idx += 1
                if b_idx >= wm_length:
                    break
        
        for b_idx in range(wm_length):
            votes_1 = votes[0, b_idx] + votes[1, b_idx] + votes[2, b_idx]
            if votes_1 >= 2:
                extracted[b_idx] = 1
            else:
                extracted[b_idx] = 0
    else:
        b_idx = 0
        for bi in range(blocks_h):
            for bj in range(blocks_w):
                if b_idx >= wm_length:
                    break
                
                for r in range(block_size):
                    for c in range(block_size):
                        block_view[r, c] = img_view[bi * block_size + r, bj * block_size + c]
                
                dwt_2d_block(block, LL, LH, HL, HH, temp_ptr, row_a_ptr, row_d_ptr, 
                            col_in_ptr, col_a_ptr, col_d_ptr, h, g, L, block_size)
                
                hl_val = hl_view[0, 0]
                lh_val = lh_view[0, 0]
                
                if hl_val > lh_val:
                    extracted[b_idx] = 1
                else:
                    extracted[b_idx] = 0
                
                b_idx += 1
            if b_idx >= wm_length:
                break


class DWT(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        """
        Встраивает биты ЦВЗ в DWT-коэффициенты блоков изображения.
        :param input_image: матрица входного изображения.
        :param watermark_bits: массив битов ЦВЗ.
        :param block_size: размер блока (8 или 16).
        :param min_difference: минимальная требуемая разность между коэффициентами HL и LH.
        :param amplification_factor: коэффициент усиления (1.1 < v < 2).
        :param wavelet_name: тип вейвлета (haar, db4, sym4).
        :param redundant: использовать избыточное встраивание (3 копии).

        :return output_image: матрица изображения с встроенным ЦВЗ.
        """
        defaults = {
            "input_image": None, 
            "watermark_bits": None, 
            "block_size": 8,
            "min_difference": 0.5, 
            "amplification_factor": 1.5,
            "wavelet_name": "haar",
            "redundant": True
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
        cdef double min_difference = kwargs["min_difference"]
        cdef double amplification_factor = kwargs["amplification_factor"]
        cdef bytes wavelet_name = kwargs["wavelet_name"].encode('ascii')
        cdef bint redundant = kwargs["redundant"]
        
        cdef double h[8], g[8]
        cdef int L
        get_wavelet_filters(wavelet_name, h, g, &L)
        
        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]
        cdef int blocks_h = H // block_size
        cdef int blocks_w = W // block_size
        cdef int half_block = block_size >> 1
        
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] watermarked_img = img_c.copy()
        cdef double[:, :] img_view = watermarked_img
        
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] LL = np.empty((half_block, half_block), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] LH = np.empty((half_block, half_block), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] HL = np.empty((half_block, half_block), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] HH = np.empty((half_block, half_block), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] block = np.empty((block_size, block_size), dtype=np.float64)
        
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] temp_1d = np.empty(block_size * block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_a = np.empty(half_block, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_d = np.empty(half_block, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_in = np.empty(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_a = np.empty(half_block, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_d = np.empty(half_block, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_out = np.empty(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_out = np.empty(block_size, dtype=np.float64)
        
        cdef double* temp_ptr = <double*>cnp.PyArray_DATA(temp_1d)
        cdef double* row_a_ptr = <double*>cnp.PyArray_DATA(row_a)
        cdef double* row_d_ptr = <double*>cnp.PyArray_DATA(row_d)
        cdef double* col_in_ptr = <double*>cnp.PyArray_DATA(col_in)
        cdef double* col_a_ptr = <double*>cnp.PyArray_DATA(col_a)
        cdef double* col_d_ptr = <double*>cnp.PyArray_DATA(col_d)
        cdef double* col_out_ptr = <double*>cnp.PyArray_DATA(col_out)
        cdef double* row_out_ptr = <double*>cnp.PyArray_DATA(row_out)
        
        _embed_core(img_view, blocks_h, blocks_w, wm_c, block_size, min_difference,
                   amplification_factor, redundant, LL, LH, HL, HH, block, temp_ptr,
                   row_a_ptr, row_d_ptr, col_in_ptr, col_a_ptr, col_d_ptr, col_out_ptr,
                   row_out_ptr, h, g, L)
        
        return watermarked_img

    @staticmethod
    def extraction(**args):
        """
        Извлекает биты ЦВЗ из DWT-коэффициентов блоков изображения.
        :param input_image: матрица изображения с ЦВЗ.
        :param num_bits: длина ЦВЗ.
        :param block_size: размер блока (8 или 16).
        :param wavelet_name: тип вейвлета (haar, db4, sym4).
        :param redundant: использовать избыточное извлечение (голосование).

        :return extracted_wm: извлечённый ЦВЗ.
        """
        defaults = {
            "input_image": None, 
            "num_bits": 0, 
            "block_size": 8,
            "wavelet_name": "haar",
            "redundant": True
        }
        kwargs = {**defaults, **args}
        
        image = kwargs.get("input_image")
        if image is None:
            image = kwargs.get("image_path")
        num_bits = kwargs.get("num_bits")
        
        if image is None or not num_bits:
            raise ValueError("Не переданы input_image/image_path или num_bits")

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(image, dtype=np.float64)
        cdef int block_size = kwargs["block_size"]
        cdef bytes wavelet_name = kwargs["wavelet_name"].encode('ascii')
        cdef bint redundant = kwargs["redundant"]
        
        cdef double h[8], g[8]
        cdef int L
        get_wavelet_filters(wavelet_name, h, g, &L)
        
        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]
        cdef int blocks_h = H // block_size
        cdef int blocks_w = W // block_size
        cdef int half_block = block_size >> 1
        
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] extracted_wm = np.zeros(num_bits, dtype=np.int32)
        cdef double[:, :] img_view = img_c
        
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] LL = np.empty((half_block, half_block), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] LH = np.empty((half_block, half_block), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] HL = np.empty((half_block, half_block), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] HH = np.empty((half_block, half_block), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] block = np.empty((block_size, block_size), dtype=np.float64)
        
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] temp_1d = np.empty(block_size * block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_a = np.empty(half_block, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_d = np.empty(half_block, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_in = np.empty(block_size, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_a = np.empty(half_block, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_d = np.empty(half_block, dtype=np.float64)
        
        cdef double* temp_ptr = <double*>cnp.PyArray_DATA(temp_1d)
        cdef double* row_a_ptr = <double*>cnp.PyArray_DATA(row_a)
        cdef double* row_d_ptr = <double*>cnp.PyArray_DATA(row_d)
        cdef double* col_in_ptr = <double*>cnp.PyArray_DATA(col_in)
        cdef double* col_a_ptr = <double*>cnp.PyArray_DATA(col_a)
        cdef double* col_d_ptr = <double*>cnp.PyArray_DATA(col_d)
        
        _extract_core(img_view, blocks_h, blocks_w, extracted_wm, num_bits, block_size,
                     redundant, LL, LH, HL, HH, block, temp_ptr, row_a_ptr, row_d_ptr,
                     col_in_ptr, col_a_ptr, col_d_ptr, h, g, L)
        
        return extracted_wm