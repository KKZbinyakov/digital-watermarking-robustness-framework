from libc.math cimport cos, sqrt, M_PI
import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy
cnp.import_array()

cdef double _C_DCT[8][8]
cdef bint _C_DCT_INITIALIZED = False

cdef void init_dct_matrix():
    """
    Инициализация базисной матрицы ДКП.

    Вычисляет ортогональную матрицу ДКП размерностью 8x8 один раз при первом вызове.
    """
    global _C_DCT, _C_DCT_INITIALIZED
    if _C_DCT_INITIALIZED:
        return
    
    cdef int k, n
    cdef double alpha, factor = sqrt(2.0 / 8.0)
    
    for k in range(8):
        alpha = 1.0 / sqrt(2.0) if k == 0 else 1.0
        for n in range(8):
            _C_DCT[k][n] = alpha * factor * cos(M_PI * (2.0 * n + 1.0) * k / 16.0)
    
    _C_DCT_INITIALIZED = True

cdef void apply_dct_8x8(const double block_img[8][8], double block_dct[8][8]):
    """
    Прямое ДКП-преобразование блока 8x8.

    Вычисляет двумерное дискретное косинусное преобразование блока пикселей.

    Args:
        block_img: блок пикселей изображения
        block_dct: выходной блок коэффициентов ДКП
        C: ортогональная матрица ДКП
    """
    cdef double temp[8][8]
    cdef int i, j, k
    cdef double sum_val

    for i in range(8):
        for j in range(8):
            sum_val = 0.0
            for k in range(8):
                sum_val += _C_DCT[i][k] * block_img[k][j]
            temp[i][j] = sum_val

    for i in range(8):
        for j in range(8):
            sum_val = 0.0
            for k in range(8):
                sum_val += temp[i][k] * _C_DCT[j][k]
            block_dct[i][j] = sum_val

cdef void apply_idct_8x8(const double block_dct[8][8], double block_idct[8][8]):
    """
    Обратное ДКП-преобразование блока 8x8.

    Вычисляет двумерное обратное дискретное косинусное преобразование блока частотных
    коэффициентов/

    Args:
        block_dct: блок коэффициентов ДКП
        block_idct: выходной блок пикселей
        C: ортогональная матрица ДКП
    """
    cdef double temp[8][8]
    cdef int i, j, k
    cdef double sum_val

    for i in range(8):
        for j in range(8):
            sum_val = 0.0
            for k in range(8):
                sum_val += _C_DCT[k][i] * block_dct[k][j]
            temp[i][j] = sum_val

    for i in range(8):
        for j in range(8):
            sum_val = 0.0
            for k in range(8):
                sum_val += temp[i][k] * _C_DCT[k][j]
            block_idct[i][j] = sum_val


cdef void get_wavelet_filters(char* name, double* h, double* g, int* L):
    """
    Инициализирует фильтры вейвлетов.

    Args:
        name: имя вейвлета
        h: указатель на массив низкочастотных коэффициентов
        g: указатель на массив высокочастотных коэффициентов
        L: указатель на длину фильтра
    """
    if name == b"haar":
        L[0] = 2
        h[0] =  0.7071067811865476; h[1] =  0.7071067811865476
        g[0] =  0.7071067811865476; g[1] = -0.7071067811865476
    elif name == b"db4":
        L[0] = 8
        h[0] =  0.2303778133088964; h[1] =  0.7148465705529154
        h[2] =  0.6308807679398587; h[3] = -0.0279837694168599
        h[4] = -0.1870348117190931; h[5] =  0.0308413818355607
        h[6] =  0.0328830116668852; h[7] = -0.0105974017850690
        g[0] = -0.0105974017850690; g[1] = -0.0328830116668852
        g[2] =  0.0308413818355607; g[3] =  0.1870348117190931
        g[4] = -0.0279837694168599; g[5] = -0.6308807679398587
        g[6] =  0.7148465705529154; g[7] = -0.2303778133088964
    elif name == b"sym4":
        L[0] = 8
        h[0] = -0.0757657147893432; h[1] = -0.0296355276459985
        h[2] =  0.4976186676323012; h[3] =  0.8037387518059161
        h[4] =  0.2978577956052774; h[5] = -0.0992195435768472
        h[6] = -0.0126039672622612; h[7] =  0.0322231006040427
        g[0] =  0.0322231006040427; g[1] =  0.0126039672622612
        g[2] = -0.0992195435768472; g[3] = -0.2978577956052774
        g[4] =  0.8037387518059161; g[5] = -0.4976186676323012
        g[6] = -0.0296355276459985; g[7] =  0.0757657147893432
    else:
        raise ValueError(f"Неизвестный вейвлет: {name}")

cdef inline void dwt_1d_haar(const double* x, double* a, double* d, int n):
    """
    Прямое 1D DWT-преобразование для Haar.

    Args:
        x: входной сигнал
        a: выходной массив аппроксимации
        d: выходной массив деталей
        n: длина входного сигнала
    """
    cdef int i, idx0, idx1
    cdef double x0, x1
    cdef double inv_sqrt2 = 0.7071067811865476
    for i in range(n >> 1):
        idx0 = i << 1
        idx1 = idx0 + 1
        if idx1 >= n: idx1 -= n
        x0 = x[idx0]
        x1 = x[idx1]
        a[i] = (x0 + x1) * inv_sqrt2
        d[i] = (x0 - x1) * inv_sqrt2

cdef inline void idwt_1d_haar(const double* a, const double* d, double* x, int n):
    """
    Обратное 1D DWT-преобразование для Haar.

    Args:
        a: массив аппроксимации
        d: массив деталей
        x: выходной восстановленный сигнал
        n: длина выходного сигнала
    """
    cdef int i, idx0, idx1
    cdef double a_val, d_val
    cdef double inv_sqrt2 = 0.7071067811865476
    for i in range(n >> 1):
        a_val = a[i] * inv_sqrt2
        d_val = d[i] * inv_sqrt2
        idx0 = i << 1
        idx1 = idx0 + 1
        if idx1 >= n: idx1 -= n
        x[idx0] = a_val + d_val
        x[idx1] = a_val - d_val

cdef inline void dwt_1d_l8(const double* x, double* a, double* d, int n, const double* h, const double* g):
    """
    Прямое 1D DWT-преобразование для Daubechies и Symlets.

    Args:
        x: входной сигнал
        a: выходной массив аппроксимации
        d: выходной массив деталей
        n: длина входного сигнала
        h: низкочастотный фильтр
        g: высокочастотный фильтр
    """
    cdef int i, k, idx
    for i in range(n >> 1):
        a[i] = 0.0
        d[i] = 0.0
        for k in range(8):
            idx = (i << 1) + k
            if idx >= n: idx -= n
            a[i] += h[k] * x[idx]
            d[i] += g[k] * x[idx]

cdef inline void idwt_1d_l8(const double* a, const double* d, double* x, int n, const double* h, const double* g):
    """
    Обратное 1D DWT-преобразование для Daubechies и Symlets.

    Args:
        a: массив аппроксимации
        d: массив деталей
        x: выходной восстановленный сигнал
        n: длина выходного сигнала
        h: низкочастотный фильтр
        g: высокочастотный фильтр
    """
    cdef int i, k, m
    cdef int half_n = n >> 1
    for i in range(n):
        x[i] = 0.0
        for k in range(half_n):
            m = i - (k << 1)
            if m < 0: m += n
            if m < 8:
                x[i] += h[m] * a[k] + g[m] * d[k]

cdef void dwt_2d_block(double[:, :] block,
                       double[:, :] LL, double[:, :] LH, double[:, :] HL, double[:, :] HH, 
                       double* temp, double* row_a, double* row_d,
                       double* col_in, double* col_a, double* col_d,
                       const double* h, const double* g, int L, int block_size):
    """
    Прямое 2D DWT-преобразование для одного блока.
    
    Args:
        block: входной блок
        LL, LH, HL, HH: выходные подполосы блока
        temp: временный буфер
        row_a, row_d: буферы строк
        col_in, col_a, col_d: буферы столбцов
        h, g: фильтры
        L: длина фильтра
        block_size: размер блока
    """
    cdef int H = block_size
    cdef int W = block_size
    cdef int half_H = H >> 1
    cdef int half_W = W >> 1
    cdef int r, c, idx
    
    if L == 2:
        for r in range(H):
            dwt_1d_haar(&block[r, 0], row_a, row_d, W)
            idx = r * W
            for c in range(half_W):
                temp[idx + c] = row_a[c]
                temp[idx + half_W + c] = row_d[c]
        for c in range(W):
            for r in range(H):
                col_in[r] = temp[r * W + c]
            dwt_1d_haar(col_in, col_a, col_d, H)
            for r in range(half_H):
                if c < half_W:
                    LL[r, c] = col_a[r]
                    HL[r, c] = col_d[r]
                else:
                    LH[r, c - half_W] = col_a[r]
                    HH[r, c - half_W] = col_d[r]
    else:
        for r in range(H):
            dwt_1d_l8(&block[r, 0], row_a, row_d, W, h, g)
            idx = r * W
            for c in range(half_W):
                temp[idx + c] = row_a[c]
                temp[idx + half_W + c] = row_d[c]
        for c in range(W):
            for r in range(H):
                col_in[r] = temp[r * W + c]
            dwt_1d_l8(col_in, col_a, col_d, H, h, g)
            for r in range(half_H):
                if c < half_W:
                    LL[r, c] = col_a[r]
                    HL[r, c] = col_d[r]
                else:
                    LH[r, c - half_W] = col_a[r]
                    HH[r, c - half_W] = col_d[r]

cdef void idwt_2d_block(double[:, :] LL, double[:, :] LH, double[:, :] HL, double[:, :] HH,
                        double[:, :] block, 
                        double* temp, double* col_a, double* col_d, double* col_out, 
                        double* row_a, double* row_d, double* row_out,
                        const double* h, const double* g, int L, int block_size):
    """
    Обратное 2D DWT-преобразование для одного блока.
    
    Args:
        LL, LH, HL, HH: входные подполосы блока
        block: выходной блок
        temp: временный буфер
        col_a, col_d, col_out: буферы столбцов
        row_a, row_d, row_out: буферы строк
        h, g: фильтры
        L: длина фильтра
        block_size: размер блока
    """
    cdef int H = block_size
    cdef int W = block_size
    cdef int half_H = H >> 1
    cdef int half_W = W >> 1
    cdef int r, c
    
    if L == 2:
        for c in range(W):
            for r in range(half_H):
                if c < half_W:
                    col_a[r] = LL[r, c]
                    col_d[r] = HL[r, c]
                else:
                    col_a[r] = LH[r, c - half_W]
                    col_d[r] = HH[r, c - half_W]
            idwt_1d_haar(col_a, col_d, col_out, H)
            for r in range(H):
                temp[r * W + c] = col_out[r]

        for r in range(H):
            for c in range(half_W):
                row_a[c] = temp[r * W + c]
                row_d[c] = temp[r * W + half_W + c]
            idwt_1d_haar(row_a, row_d, row_out, W)
            memcpy(&block[r, 0], row_out, W * sizeof(double))
    else:
        for c in range(W):
            for r in range(half_H):
                if c < half_W:
                    col_a[r] = LL[r, c]
                    col_d[r] = HL[r, c]
                else:
                    col_a[r] = LH[r, c - half_W]
                    col_d[r] = HH[r, c - half_W]
            idwt_1d_l8(col_a, col_d, col_out, H, h, g)
            for r in range(H):
                temp[r * W + c] = col_out[r]
        for r in range(H):
            for c in range(half_W):
                row_a[c] = temp[r * W + c]
                row_d[c] = temp[r * W + half_W + c]
            idwt_1d_l8(row_a, row_d, row_out, W, h, g)
            memcpy(&block[r, 0], row_out, W * sizeof(double))