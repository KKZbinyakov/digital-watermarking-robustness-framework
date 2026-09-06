"""
Метод DWT (Discrete Wavelet Transform) — в подполосы LL, LH, HL, HH (Haar, Daubechies, Symlets).
https://ictactjournals.in/paper/IJIVP_V6_I2_paper_5_1133_1136.pdf
"""

import numpy as np
cimport numpy as cnp

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
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
        min_difference: минимальная разность между коэффициентами HL и LH
            в единицах DWT-коэффициента.
        amplification_factor: коэффициент усиления.
        redundant: использовать избыточное встраивание трёх копий
            в непересекающиеся области блоков изображения.
        LL, LH, HL, HH: буферы подполос.
        block: буфер блока.
        temp_ptr, row_a_ptr, row_d_ptr, col_in_ptr, col_a_ptr, col_d_ptr, col_out_ptr, row_out_ptr: указатели на буферы.
        h, g: фильтры.
        L: длина фильтра.
    """
    cdef int wm_len = watermark.shape[0]
    cdef int copies = 3 if redundant else 1
    cdef int capacity = blocks_h * blocks_w
    cdef int copy_stride = capacity // copies
    cdef int bi, bj, r, c, copy_idx, b_idx, flat_idx
    cdef double hl_val, lh_val, diff
    cdef double[:, :] hl_view = HL
    cdef double[:, :] lh_view = LH
    cdef double[:, :] block_view = block

    for copy_idx in range(copies):
        for b_idx in range(wm_len):
            flat_idx = copy_idx * copy_stride + b_idx
            bi = flat_idx // blocks_w
            bj = flat_idx % blocks_w

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
        redundant: использовать избыточное извлечение из трёх
            непересекающихся областей с голосованием по большинству.
        LL, LH, HL, HH: буферы подполос.
        block: буфер блока.
        temp_ptr, row_a_ptr, row_d_ptr, col_in_ptr, col_a_ptr, col_d_ptr: указатели на буферы.
        h, g: фильтры.
        L: длина фильтра.
    """
    cdef int capacity = blocks_h * blocks_w
    cdef int copies = 3 if redundant else 1
    cdef int copy_stride = capacity // copies
    cdef int bi, bj, r, c, copy_idx, b_idx, flat_idx
    cdef double hl_val, lh_val
    cdef int votes_1
    cdef double[:, :] hl_view = HL
    cdef double[:, :] lh_view = LH
    cdef double[:, :] block_view = block
    
    cdef cnp.ndarray[cnp.int32_t, ndim=2, mode='c'] votes = np.zeros((3, wm_length), dtype=np.int32)
    
    if redundant:
        for copy_idx in range(3):
            for b_idx in range(wm_length):
                flat_idx = copy_idx * copy_stride + b_idx
                bi = flat_idx // blocks_w
                bj = flat_idx % blocks_w

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
        :param input_image: RGB-изображение uint8 формы (H, W, 3).
        :param watermark_bits: массив uint8 из значений 0/1.
        :param block_size: размер блока (8 или 16).
        :param min_difference: минимальная требуемая разность между коэффициентами HL и LH
            в единицах DWT-коэффициента.
        :param amplification_factor: коэффициент усиления (1.1 < v < 2).
        :param wavelet_name: тип вейвлета (haar, db4, sym4).
        :param redundant: использовать избыточное встраивание: три копии ЦВЗ
            размещаются в трёх непересекающихся областях блоков.

        :return output_image: RGB-изображение uint8 формы (H, W, 3) со встроенным ЦВЗ.
        """
        defaults = {
                    "input_image": None, 
                    "watermark_bits": None, 
                    "block_size": 8,
                    "min_difference": 20.0, 
                    "amplification_factor": 1.5,
                    "wavelet_name": "haar",
                    "redundant": True
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
        cdef double min_difference = args["min_difference"]
        cdef double amplification_factor = args["amplification_factor"]
        cdef bytes wavelet_name = args["wavelet_name"].encode('ascii')
        cdef bint redundant = args["redundant"]
        
        cdef double h[8], g[8]
        cdef int L
        get_wavelet_filters(wavelet_name, h, g, &L)
        
        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]
        cdef int blocks_h = H // block_size
        cdef int blocks_w = W // block_size
        cdef int capacity = blocks_h * blocks_w
        cdef int wm_len = wm_c.shape[0]
        cdef int copies = 3 if redundant else 1
        cdef int required_blocks = wm_len * copies

        if required_blocks > capacity:
            raise ValueError(
                f"Not enough capacity: need {required_blocks} blocks "
                f"for {copies} watermark copy/copies, available {capacity}."
            )
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
        
        watermarked_y = np.asarray(watermarked_img, dtype=np.float64)
        delta_y = watermarked_y - input_y
        output_rgb = rgb_c.astype(np.float64) + delta_y[:, :, None]
        return np.ascontiguousarray(
            np.clip(np.rint(output_rgb), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def extraction(**args):
        """
        Извлекает биты ЦВЗ из DWT-коэффициентов блоков изображения.
        :param input_image: RGB-изображение uint8 формы (H, W, 3) с ЦВЗ.
        :param num_bits: длина ЦВЗ.
        :param block_size: размер блока (8 или 16).
        :param wavelet_name: тип вейвлета (haar, db4, sym4).
        :param redundant: использовать избыточное извлечение из трёх
            непересекающихся копий с голосованием по большинству.

        :return extracted_wm: извлечённый int8-массив из значений 0/1.
        """
        defaults = {
                    "input_image": None, 
                    "num_bits": 0, 
                    "block_size": 8,
                    "wavelet_name": "haar",
                    "redundant": True
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
        cdef int block_size = int(args["block_size"])
        cdef bytes wavelet_name = args["wavelet_name"].encode('ascii')
        cdef bint redundant = args["redundant"]
        
        cdef double h[8], g[8]
        cdef int L
        get_wavelet_filters(wavelet_name, h, g, &L)
        
        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]
        cdef int blocks_h = H // block_size
        cdef int blocks_w = W // block_size
        cdef int capacity = blocks_h * blocks_w
        cdef int copies = 3 if redundant else 1
        cdef int required_blocks = num_bits * copies

        if required_blocks > capacity:
            raise ValueError(
                f"Cannot extract {num_bits} bits with {copies} copy/copies: "
                f"need {required_blocks} blocks, capacity is {capacity}."
            )
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
        
        if np.any((extracted_wm != 0) & (extracted_wm != 1)):
            raise RuntimeError("dwt extraction produced a non-binary watermark")
        return np.ascontiguousarray(extracted_wm, dtype=np.int8)