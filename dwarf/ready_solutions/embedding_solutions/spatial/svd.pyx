"""
Метод SVD (Singular Value Decomposition) — модификация сингулярных чисел.
https://www.mdpi.com/1999-5903/9/3/45
"""

import numpy as np
import warnings
cimport numpy as cnp
from cython.parallel cimport prange # type: ignore

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils cimport (
    embed_block, extract_block, N_SVD
)

cnp.import_array()

cdef void _embed_core(double[:, ::1] out, int[::1] watermark, int used, int nbw, double margin) noexcept nogil:
    """
    Ядро встраивания ЦВЗ.
    
    Args:
        out: выходное изображение.
        watermark: биты ЦВЗ.
        used: количество используемых блоков.
        nbw: количество блоков по ширине.
        margin: шаг квантования.
    """
    cdef Py_ssize_t b
    cdef int L = watermark.shape[0]
    
    for b in prange(used, nogil=True, schedule='static'):
        embed_block(out,
                     <int>(b // nbw), <int>(b % nbw),
                     watermark[b % L],
                     margin)

cdef void _extract_core(double[:, ::1] image, int[::1] raw, int used, int nbw, double margin) noexcept nogil:
    """
    Ядро извлечения ЦВЗ.
    
    Args:
        image: входное изображение.
        raw: выходной массив сырых бит.
        used: количество используемых блоков.
        nbw: количество блоков по ширине.
        margin: шаг квантования.
    """
    cdef Py_ssize_t b
    
    for b in prange(used, nogil=True, schedule='static'):
        raw[b] = extract_block(image,
                                <int>(b // nbw), <int>(b % nbw),
                                margin)

def normalize_redundancy(int redundancy, str func_name):
    """
    Приведение redundancy к нечётному значению.

    Args:
        redundancy: запрошенное число копий каждого бита.
        func_name: имя вызывающей функции.

    Returns:
        redundancy: скорректированное значение redundancy.
    """
    if redundancy % 2 == 0:
        warnings.warn(
            f"{func_name}: redundancy={redundancy} чётное, мажоритарное голосование "
            f"даёт ничью на половине голосов и смещает результат в пользу нуля. "
            f"Значение понижено до {redundancy - 1}. Одно и то же значение redundancy "
            f"должно передаваться и при встраивании, и при извлечении.",
            UserWarning,
            stacklevel=1,
        )
        return redundancy - 1
    return redundancy


class SVD(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        """
        Встраивает биты ЦВЗ в сингулярные числа блоков 4x4 методом QIM.
        :param input_image: матрица входного изображения (канал яркости Y).
        :param watermark_bits: массив битов ЦВЗ.
        :param margin: шаг квантования. Больше - устойчивее, но заметнее.
        :param redundancy: сколько блоков тратится на один бит.

        :return output_image: матрица изображения с встроенным ЦВЗ.
        """
        defaults = {
            "input_image": None,
            "watermark_bits": None,
            "margin": 40.0,
            "redundancy": 1
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
        
        cdef double margin = kwargs["margin"]
        cdef int redundancy = kwargs["redundancy"]
        
        if margin <= 0.0:
            raise ValueError("margin (шаг квантования) должен быть положительным.")
        if redundancy < 1:
            raise ValueError("redundancy должен быть >= 1.")
            
        redundancy = normalize_redundancy(redundancy, "SVD.embedding")
        
        cdef Py_ssize_t H = img_c.shape[0]
        cdef Py_ssize_t W = img_c.shape[1]
        cdef Py_ssize_t L = wm_c.shape[0]
        
        cdef Py_ssize_t nb_h = H // N_SVD
        cdef Py_ssize_t nb_w = W // N_SVD
        cdef Py_ssize_t capacity = nb_h * nb_w
        cdef Py_ssize_t used = L * redundancy
        
        if used > capacity:
            raise ValueError(
                f"Не хватает ёмкости: нужно {used} блоков, доступно {capacity}. "
                f"Уменьшите длину ЦВЗ или redundancy."
            )
            
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] out_np = img_c.copy()
        cdef double[:, ::1] out_mv = out_np
        cdef int[::1] wm_mv = wm_c
        cdef int nbw = <int>nb_w
        
        _embed_core(out_mv, wm_mv, <int>used, nbw, margin)
        
        return out_np

    @staticmethod
    def extraction(**args):
        """
        Извлекает биты ЦВЗ из сингулярных чисел блоков 4x4 методом QIM.
        :param input_image: матрица изображения с ЦВЗ (канал яркости Y).
        :param num_bits: длина ЦВЗ.
        :param margin: шаг квантования.
        :param redundancy: число копий каждого бита.

        :return extracted_wm: извлечённые биты ЦВЗ.
        """
        defaults = {
            "input_image": None,
            "num_bits": 0,
            "margin": 40.0,
            "redundancy": 1
        }
        kwargs = {**defaults, **args}
        
        image = kwargs.get("input_image")
        if image is None:
            image = kwargs.get("image_path")
        num_bits = kwargs.get("num_bits")
        
        if image is None or not num_bits:
            raise ValueError("Не переданы input_image/image_path или num_bits")

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(image, dtype=np.float64)
        
        cdef double margin = kwargs["margin"]
        cdef int redundancy = kwargs["redundancy"]
        cdef int wm_length = num_bits
        
        if margin <= 0.0:
            raise ValueError("margin (шаг квантования) должен быть положительным.")
        if redundancy < 1:
            raise ValueError("redundancy должен быть >= 1.")
            
        redundancy = normalize_redundancy(redundancy, "SVD.extraction")
        
        cdef Py_ssize_t H = img_c.shape[0]
        cdef Py_ssize_t W = img_c.shape[1]
        
        cdef Py_ssize_t nb_h = H // N_SVD
        cdef Py_ssize_t nb_w = W // N_SVD
        cdef Py_ssize_t used = wm_length * redundancy
        
        if used > nb_h * nb_w:
            raise ValueError("wm_length * redundancy превышает число блоков 4x4.")
            
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] raw_np = np.empty(used, dtype=np.int32)
        cdef int[::1] raw_mv = raw_np
        cdef double[:, ::1] img_mv = img_c
        cdef int nbw = <int>nb_w
        
        _extract_core(img_mv, raw_mv, <int>used, nbw, margin)
        
        if redundancy == 1:
            return raw_np
            
        votes = raw_np.reshape(redundancy, wm_length).sum(axis=0)
        return (votes * 2 > redundancy).astype(np.int32)

    @staticmethod
    def capacity_bits(int height, int width, int redundancy=1):
        """
        Возвращает максимальную длину ЦВЗ (в битах) для изображения данного размера.
        :param height: высота изображения.
        :param width: ширина изображения.
        :param redundancy: число копий каждого бита.

        :return: максимальная длина ЦВЗ.
        """
        if redundancy < 1:
            raise ValueError("redundancy должен быть >= 1.")
        return (height // N_SVD) * (width // N_SVD) // redundancy