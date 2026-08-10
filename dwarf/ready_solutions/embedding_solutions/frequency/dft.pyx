"""
Метод DFT (Discrete Fourier Transform) с логарифмической поляризацией Fourier-Mellin
https://cecs.uci.edu/~papers/icme05/defevent/papers/cr1102.pdf?spm=a2ty_o01.29997173.0.0.ee5755fbuHKEmE&file=cr1102.pdf
(с небольшими изменениями)
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport log

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    build_cell_map, cell_mean_sync, apply_gain
)

cnp.import_array()

BARKER13 = np.array([+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1],
                    dtype=np.float64)

cdef void _embed_core(double complex[:, ::1] F_mv, int[:, ::1] cell_mv, 
                      int n_rho, int n_theta, int n_cells, int n_sync,
                      double[::1] target_mv, int n_iter, double max_gain):
    """
    Ядро встраивания ЦВЗ в спектр.
    
    Args:
        F_mv: представление спектра.
        cell_mv: карта ячеек.
        n_rho: число подколец.
        n_theta: число угловых секторов.
        n_cells: общее число ячеек.
        n_sync: число символов маркера.
        target_mv: целевые разности.
        n_iter: число итераций.
        max_gain: максимальное усиление.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] gain = np.empty(n_cells, dtype=np.float64)
    cdef double[::1] gain_mv = gain
    cdef double lg = log(max_gain)
    cdef int it
    cdef cnp.ndarray S, d, adj, G
    cdef cnp.ndarray[cnp.float64_t, ndim=1] target = np.asarray(target_mv)

    for it in range(n_iter):
        S = cell_mean_sync(F_mv, cell_mv, n_rho, n_theta)
        d = S[:, 0::2] - S[:, 1::2]
        adj = np.clip(0.5 * (target[None, :] - d), -lg, lg)
        G = np.empty((n_rho, n_theta), dtype=np.float64)
        G[:, 0::2] = np.exp(adj)
        G[:, 1::2] = np.exp(-adj)
        gain[:] = G.reshape(-1)
        with nogil:
            apply_gain(F_mv, cell_mv, gain_mv)

cdef void _extract_core(double complex[:, ::1] F_mv, int[:, ::1] cellm, 
                        int n_rho, int n_fine, int n_sync, int wm_length,
                        int OS, bint search_rotation, int[::1] bits_mv,
                        double[::1] b_mv):
    """
    Ядро извлечения ЦВЗ из спектра.
    
    Args:
        F_mv: представление спектра.
        cellm: карта ячеек.
        n_rho: число подколец.
        n_fine: число угловых секторов с оверсэмплингом.
        n_sync: число символов маркера.
        wm_length: длина ЦВЗ.
        OS: коэффициент оверсэмплинга.
        search_rotation: искать ли поворот.
        bits_mv: выходной массив бит.
        b_mv: массив кода Баркера.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] A = np.ascontiguousarray(cell_mean_sync(F_mv, cellm, n_rho, n_fine).sum(axis=0))
    cdef double[::1] A_mv = A
    
    cdef int tau, best = 0, j, p, m0, m1, i
    cdef double c, bestc = -1e300, sa, sb

    if search_rotation and n_sync > 0:
        with nogil:
            for tau in range(n_fine):
                c = 0.0
                for j in range(n_sync):
                    sa = 0.0
                    sb = 0.0
                    m0 = (2 * j) * OS + tau
                    m1 = (2 * j + 1) * OS + tau
                    for p in range(OS):
                        sa += A_mv[(m0 + p) % n_fine]
                        sb += A_mv[(m1 + p) % n_fine]
                    c += b_mv[j] * (sa - sb)
                if c > bestc:
                    bestc = c
                    best = tau

    with nogil:
        for i in range(wm_length):
            sa = 0.0
            sb = 0.0
            m0 = (2 * (n_sync + i)) * OS + best
            m1 = (2 * (n_sync + i) + 1) * OS + best
            for p in range(OS):
                sa += A_mv[(m0 + p) % n_fine]
                sb += A_mv[(m1 + p) % n_fine]
            bits_mv[i] = 1 if sa > sb else 0


class DFT(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        """
        Встраивает биты ЦВЗ в амплитудный спектр изображения с использованием лог-полярной сетки.
        :param input_image: матрица входного изображения (канал яркости Y).
        :param watermark_bits: массив битов ЦВЗ.
        :param margin: целевая разность средних ln|F| внутри пары секторов.
        :param n_rho: число подколец.
        :param r_min: внутренняя граница рабочего кольца.
        :param r_max: внешняя граница рабочего кольца.
        :param n_sync: число символов маркера (кода Баркера).
        :param sync_boost: во сколько раз амплитуда символов маркера больше margin.
        :param n_iter: число итераций уточнения усиления.
        :param max_gain: ограничение на коэффициент усиления ячейки.

        :return output_image: матрица изображения с встроенным ЦВЗ.
        """
        defaults = {
                    "input_image": None,
                    "watermark_bits": None,
                    "margin": 0.40,
                    "n_rho": 4,
                    "r_min": 0.10,
                    "r_max": 0.42,
                    "n_sync": 13,
                    "sync_boost": 1.5,
                    "n_iter": 3,
                    "max_gain": 3.0
                }
        args = {**defaults, **args}
        
        image = args.get["input_image"]
        watermark = args.get["watermark_bits"]
        
        if image is None or watermark is None:
            raise ValueError("input_image/image_path or watermark_bits not given")

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(image, dtype=np.float64)
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] wm_c = np.ascontiguousarray(watermark, dtype=np.int32)
        
        cdef double margin = args["margin"]
        cdef int n_rho = int(args["n_rho"])
        cdef double r_min = args["r_min"]
        cdef double r_max = args["r_max"]
        cdef int n_sync = int(args["n_sync"])
        cdef double sync_boost = args["sync_boost"]
        cdef int n_iter = int(args["n_iter"])
        cdef double max_gain = args["max_gain"]
        
        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]
        cdef int L = wm_c.shape[0]
        
        if L <= 0:
            raise ValueError("Empty watermark")
        if not (0.0 < r_min < r_max < 0.5):
            raise ValueError("Needed 0 < r_min < r_max < 0.5")
        if n_rho < 1:
            raise ValueError("n_rho must be >= 1")
            
        cdef int n_sec = n_sync + L
        cdef int n_theta = 2 * n_sec
        cdef int n_cells = n_rho * n_theta

        cdef cnp.ndarray[cnp.complex128_t, ndim=2, mode='c'] F = \
            np.ascontiguousarray(np.fft.fft2(img_c), dtype=np.complex128)
        cdef double complex[:, ::1] F_mv = F

        cdef cnp.ndarray[cnp.int32_t, ndim=2, mode='c'] cellm = \
            build_cell_map(H, W, n_rho, n_theta, r_min, r_max)
        cdef int[:, ::1] cell_mv = cellm

        sym = np.empty(n_sec, dtype=np.float64)
        if n_sync > 0:
            sym[:n_sync] = BARKER13[np.arange(n_sync) % 13] * sync_boost
        sym[n_sync:] = np.where(np.asarray(wm_c) > 0, 1.0, -1.0)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] target = margin * sym
        cdef double[::1] target_mv = target

        _embed_core(F_mv, cell_mv, n_rho, n_theta, n_cells, n_sync, target_mv, n_iter, max_gain)

        return np.ascontiguousarray(np.fft.ifft2(F).real, dtype=np.float64)

    @staticmethod
    def extraction(**args):
        """
        Извлекает биты ЦВЗ из амплитудного спектра изображения.
        :param input_image: матрица изображения с ЦВЗ (канал яркости Y).
        :param num_bits: длина ЦВЗ.
        :param n_rho: число подколец.
        :param r_min: внутренняя граница кольца.
        :param r_max: внешняя граница кольца.
        :param n_sync: число символов маркера.
        :param search_rotation: оценивать ли циклический сдвиг секторов.
        :param oversampling: коэффициент оверсэмплинга по углу.

        :return extracted_wm: извлечённый ЦВЗ.
        """
        defaults = {
                    "input_image": None,
                    "num_bits": 0,
                    "n_rho": 4,
                    "r_min": 0.10,
                    "r_max": 0.42,
                    "n_sync": 13,
                    "search_rotation": True,
                    "oversampling": 4
                }
        args = {**defaults, **args}
        
        image = args.get["input_image"]
        num_bits = args.get["num_bits"]
        
        if image is None or not num_bits:
            raise ValueError("input_image/image_path or watermark_bits not given")

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(image, dtype=np.float64)
        cdef int wm_length = num_bits
        
        cdef int n_rho = int(args["n_rho"])
        cdef double r_min = args["r_min"]
        cdef double r_max = args["r_max"]
        cdef int n_sync = int(args["n_sync"])
        cdef bint search_rotation = args["search_rotation"]
        cdef int oversampling = int(args["oversampling"])
        
        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]
        cdef int n_sec = n_sync + wm_length
        cdef int OS = oversampling if (search_rotation and n_sync > 0) else 1
        cdef int n_fine = 2 * n_sec * OS

        if OS < 1:
            raise ValueError("oversampling must be >= 1")

        cdef cnp.ndarray[cnp.complex128_t, ndim=2, mode='c'] F = \
            np.ascontiguousarray(np.fft.fft2(img_c), dtype=np.complex128)
        cdef double complex[:, ::1] F_mv = F

        cdef cnp.ndarray[cnp.int32_t, ndim=2, mode='c'] cellm = \
            build_cell_map(H, W, n_rho, n_fine, r_min, r_max)
            
        cdef cnp.ndarray[cnp.float64_t, ndim=1] bark = \
            np.ascontiguousarray(BARKER13[np.arange(n_sync if n_sync > 0 else 1) % 13])
        cdef double[::1] b_mv = bark
        
        cdef cnp.ndarray[cnp.int32_t, ndim=1] bits = np.empty(wm_length, dtype=np.int32)
        cdef int[::1] bits_mv = bits

        _extract_core(F_mv, cellm, n_rho, n_fine, n_sync, wm_length, OS, search_rotation, bits_mv, b_mv)

        return bits