from libc.math cimport cos, sqrt, atan2, log, fabs, floor, ceil, M_PI
import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy
cnp.import_array()
import warnings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    N_SVD, NN_SVD, MAX_SWEEPS, MAX_POWER_ITER
)


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
    коэффициентов.

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
        raise ValueError(f"Unknown wavelet: {name}")

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


cdef cnp.ndarray build_cell_map(int H, int W, int n_rho, int n_theta,
                                 double r1, double r2):
    """
    Построение карты ячеек лог-полярной сетки после преобразования Фурье.

    Args:
        H: высота изображения в пикселях
        W: ширина изображения в пикселях
        n_rho: число подколец, равномерных по log(r)
        n_theta: число угловых секторов на диапазон [0, pi)
        r1: внутренняя граница кольца, периодов на пиксель (0 < r1 < r2)
        r2: внешняя граница кольца, периодов на пиксель (r2 < 0.5)

    Returns:
        out: итоговая карта
    """
    cdef cnp.ndarray[cnp.int32_t, ndim=2, mode='c'] out = np.empty((H, W), dtype=np.int32)
    cdef int[:, ::1] cell = out

    cdef double log_r1 = log(r1)
    cdef double inv_drho = n_rho / (log(r2) - log_r1)
    cdef double inv_dtheta = n_theta / M_PI
    cdef double inv_H = 1.0 / H
    cdef double inv_W = 1.0 / W

    cdef int i, j, k, m
    cdef int h_half = H // 2
    cdef int w_half = W // 2
    cdef double u, v, r, th, rsq
    cdef double r1sq = r1 * r1
    cdef double r2sq = r2 * r2

    with nogil:
        for i in range(H):
            u = <double>(i if i <= h_half else i - H) * inv_H
            for j in range(W):
                v = <double>(j if j <= w_half else j - W) * inv_W

                rsq = u * u + v * v
                if rsq < r1sq or rsq >= r2sq:
                    cell[i, j] = -1
                    continue

                r = sqrt(rsq)

                k = <int>((log(r) - log_r1) * inv_drho)
                if k < 0:
                    k = 0
                elif k >= n_rho:
                    k = n_rho - 1

                th = atan2(v, u)
                if th < 0.0:
                    th += M_PI
                if th >= M_PI:
                    th -= M_PI
                m = <int>(th * inv_dtheta)
                if m < 0:
                    m = 0
                elif m >= n_theta:
                    m = n_theta - 1

                cell[i, j] = k * n_theta + m
    return out

cdef void accumulate(double complex[:, ::1] F, int[:, ::1] cell,
                      double[::1] sum_log, cnp.int64_t[::1] count) noexcept nogil:
    """
    Накопление статистик по ячейкам.

    Args:
        F: спектр изображения в "несдвинутой" раскладке np.fft.fft2
        cell: карта ячеек той же формы, результат build_cell_map()
        sum_log: накопитель сумм для отсчётов внутри кольца
        count: накопитель числа отсчётов внутри кольца
    """
    cdef Py_ssize_t H = F.shape[0], W = F.shape[1], i, j
    cdef int c
    cdef double re, im

    for i in range(H):
        for j in range(W):
            c = cell[i, j]
            if c < 0:
                continue
            re = F[i, j].real
            im = F[i, j].imag
            sum_log[c] += log(1.0 + sqrt(re * re + im * im))
            count[c] += 1

cdef void apply_gain(double complex[:, ::1] F, int[:, ::1] cell,
                     double[::1] gain) noexcept nogil:
    """
    Умножение коэффициентов спектра на коэффициент усиления своей ячейки.

    Args:
        F: спектр в "несдвинутой" раскладке np.fft.fft2
        cell: карта ячеек той же формы, результат build_cell_map()
        gain: коэффициент усиления для каждой ячейки
    """
    cdef Py_ssize_t H = F.shape[0], W = F.shape[1], i, j
    cdef int c
    cdef double g

    for i in range(H):
        for j in range(W):
            c = cell[i, j]
            if c < 0:
                continue
            g = gain[c]
            F[i, j].real = F[i, j].real * g
            F[i, j].imag = F[i, j].imag * g

cdef cnp.ndarray cell_mean_sync(double complex[:, ::1] F, int[:, ::1] cell,
                             int n_rho, int n_theta):
    """
    Подсчёт средних ln(1 + |F|) по ячейкам лог-полярной сетки, обработка пустых ячеек.

    Args:
        F: спектр в "несдвинутой" раскладке np.fft.fft2
        cell: карта ячеек той же формы, результат build_cell_map()
        n_rho: число подколец
        n_theta: число угловых секторов

    Returns:
        S: среднее ln(1 + |F|) по каждой ячейке, для пустых ячеек - средний уровень их подкольца
    """
    cdef int n_cells = n_rho * n_theta
    cdef cnp.ndarray[cnp.float64_t, ndim=1] s = np.zeros(n_cells, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] c = np.zeros(n_cells, dtype=np.int64)
    cdef double[::1] s_mv = s
    cdef cnp.int64_t[::1] c_mv = c
    cdef int t

    with nogil:
        accumulate(F, cell, s_mv, c_mv)
        for t in range(n_cells):
            if c_mv[t] > 0:
                s_mv[t] = s_mv[t] / c_mv[t]

    S = s.reshape(n_rho, n_theta)
    C = c.reshape(n_rho, n_theta)
    if (C == 0).any():
        occ = C > 0
        ring_mean = np.where(occ.any(axis=1),
                             (S * occ).sum(axis=1) / np.maximum(occ.sum(axis=1), 1),
                             0.0)
        S = np.where(occ, S, ring_mean[:, None])
    return np.ascontiguousarray(S)


cdef void svd_jacobi(double *A, double *V, double *sv, bint want_v) noexcept nogil:
    """
    Односторонний (правосторонний) метод Якоби.

    Args:
        A: входная матрица N x N.
        V: буфер под правые сингулярные векторы.
        sv: буфер под сингулярные числа.
        want_v: флаг, накапливать ли V (по сути - встраивание или извлечение).
    """
    cdef int p, q, i, sweep
    cdef double alpha, beta, gamma, zeta, t, c, s, ap, aq
    cdef double off

    if want_v:
        for i in range(NN_SVD):
            V[i] = 0.0
        for i in range(N_SVD):
            V[i * N_SVD + i] = 1.0

    for sweep in range(MAX_SWEEPS):
        off = 0.0

        for p in range(N_SVD - 1):
            for q in range(p + 1, N_SVD):

                alpha = 0.0
                beta = 0.0
                gamma = 0.0
                for i in range(N_SVD):
                    ap = A[i * N_SVD + p]
                    aq = A[i * N_SVD + q]
                    alpha += ap * ap
                    beta += aq * aq
                    gamma += ap * aq

                if gamma == 0.0:
                    continue

                off += gamma * gamma / (alpha * beta + 1e-300)

                if fabs(gamma) <= 1e-15 * sqrt(alpha * beta):
                    continue

                zeta = (beta - alpha) / (2.0 * gamma)
                if zeta >= 0.0:
                    t = 1.0 / (zeta + sqrt(1.0 + zeta * zeta))
                else:
                    t = -1.0 / (-zeta + sqrt(1.0 + zeta * zeta))
                c = 1.0 / sqrt(1.0 + t * t)
                s = c * t

                for i in range(N_SVD):
                    ap = A[i * N_SVD + p]
                    aq = A[i * N_SVD + q]
                    A[i * N_SVD + p] = c * ap - s * aq
                    A[i * N_SVD + q] = s * ap + c * aq

                if want_v:
                    for i in range(N_SVD):
                        ap = V[i * N_SVD + p]
                        aq = V[i * N_SVD + q]
                        V[i * N_SVD + p] = c * ap - s * aq
                        V[i * N_SVD + q] = s * ap + c * aq

        if off <= 1e-30:
            break

    for q in range(N_SVD):
        alpha = 0.0
        for i in range(N_SVD):
            alpha += A[i * N_SVD + q] * A[i * N_SVD + q]
        sv[q] = sqrt(alpha)

cdef inline void top2(double *sv, int *idx1, double *s1, double *s2) noexcept nogil:
    """
    Поиск 2 наибольших сингулярных чисел (с определением позиции наибольшего).

    Args:
        sv: сингулярные числа в произвольном порядке,
            результат работы svd_jacobi.
        idx1: позиция наибольшего числа в массиве sv.
        s1: наибольшее сингулярное число.
        s2: второе по величине сингулярное число.
    """
    cdef int i, k1 = 0
    cdef double m1 = sv[0]
    cdef double m2 = -1.0

    for i in range(1, N_SVD):
        if sv[i] > m1:
            m2 = m1
            m1 = sv[i]
            k1 = i
        elif sv[i] > m2:
            m2 = sv[i]

    idx1[0] = k1
    s1[0] = m1
    s2[0] = m2

cdef inline double qim_embed(double sigma, int bit, double delta) noexcept nogil:
    """
    Встраивание бита ЦВЗ методом QIM

    Args:
        sigma: текущее старшее сингулярное число блока.
        bit: встраиваемый бит ЦВЗ.
        delta: шаг квантования.

    Returns:
        новое значение старшего сингулярного числа.
    """
    cdef double T1, T2, c1, c2
    cdef int k1
    
    if bit == 1:
        T1 = 0.5 * delta
        T2 = -1.5 * delta
    else:
        T1 = -0.5 * delta
        T2 = 1.5 * delta
        
    k1 = <int>floor(ceil(sigma / delta) / 2.0)
    c1 = 2.0 * k1 * delta + T1
    c2 = 2.0 * k1 * delta + T2
    
    if fabs(sigma - c2) < fabs(sigma - c1):
        return c2
    return c1

cdef inline int qim_extract(double sigma, double delta) noexcept nogil:
    """
    Извлечение бита ЦВЗ методом QIM.

    Args:
        sigma: наблюдаемое старшее сингулярное число блока.
        delta: шаг квантования.

    Returns:
        извлечённый бит ЦВЗ.
    """
    cdef double r = sigma / delta
    cdef int k = <int>ceil(r)
    return k % 2

cdef void power_iteration(double *A, double *u, double *v, double *sigma) noexcept nogil:
    """
    Вычисление наибольшего сингулярного числа и соответствующих векторов.
    
    Args:
        A: входная матрица N x N.
        u: буфер под левый сингулярный вектор (N).
        v: буфер под правый сингулярный вектор (N).
        sigma: буфер под наибольшее сингулярное число.
    """
    cdef int i, j, iter_idx
    cdef double sum_val, norm_val
    cdef double v_old[N_SVD]
    cdef double u_tmp[N_SVD]
    cdef double v_tmp[N_SVD]
    
    for i in range(N_SVD):
        v[i] = 1.0 / sqrt(<double>N_SVD)
        
    for iter_idx in range(MAX_POWER_ITER):
        for i in range(N_SVD):
            v_old[i] = v[i]
            
        for i in range(N_SVD):
            sum_val = 0.0
            for j in range(N_SVD):
                sum_val += A[i * N_SVD + j] * v[j]
            u_tmp[i] = sum_val
            
        norm_val = 0.0
        for i in range(N_SVD):
            norm_val += u_tmp[i] * u_tmp[i]
        norm_val = sqrt(norm_val)
        if norm_val < 1e-12:
            sigma[0] = 0.0
            return
        for i in range(N_SVD):
            u[i] = u_tmp[i] / norm_val
            
        for j in range(N_SVD):
            sum_val = 0.0
            for i in range(N_SVD):
                sum_val += A[i * N_SVD + j] * u[i]
            v_tmp[j] = sum_val
            
        norm_val = 0.0
        for j in range(N_SVD):
            norm_val += v_tmp[j] * v_tmp[j]
        sigma[0] = sqrt(norm_val)
        if sigma[0] < 1e-12:
            return
        for j in range(N_SVD):
            v[j] = v_tmp[j] / sigma[0]
            
        sum_val = 0.0
        for i in range(N_SVD):
            sum_val += fabs(v[i] - v_old[i])
        if sum_val < 1e-6:
            break

cdef inline double power_iteration_sigma(double *A) noexcept nogil:
    """
    Вычисление наибольшего сингулярного числа (матричной 2-нормы).
    
    Args:
        A: входная матрица N x N.
        
    Returns:
        наибольшее сингулярное число.
    """
    cdef int i, j, iter_idx
    cdef double sum_val, norm_val
    cdef double v[N_SVD]
    cdef double u_tmp[N_SVD]
    cdef double v_tmp[N_SVD]
    
    for i in range(N_SVD):
        v[i] = 1.0 / sqrt(<double>N_SVD)
        
    for iter_idx in range(MAX_POWER_ITER):
        for i in range(N_SVD):
            sum_val = 0.0
            for j in range(N_SVD):
                sum_val += A[i * N_SVD + j] * v[j]
            u_tmp[i] = sum_val
            
        norm_val = 0.0
        for i in range(N_SVD):
            norm_val += u_tmp[i] * u_tmp[i]
        norm_val = sqrt(norm_val)
        if norm_val < 1e-12:
            return 0.0
        for i in range(N_SVD):
            u_tmp[i] /= norm_val
            
        for j in range(N_SVD):
            sum_val = 0.0
            for i in range(N_SVD):
                sum_val += A[i * N_SVD + j] * u_tmp[i]
            v_tmp[j] = sum_val
            
        norm_val = 0.0
        for j in range(N_SVD):
            norm_val += v_tmp[j] * v_tmp[j]
        norm_val = sqrt(norm_val)
        if norm_val < 1e-12:
            return 0.0
        for j in range(N_SVD):
            v[j] = v_tmp[j] / norm_val
            
    return norm_val

cdef void embed_block(double[:, ::1] img, int by, int bx, int bit,
                       double delta) noexcept nogil:
    """
    Встраивание одного бита ЦВЗ в один блок 4x4 изображения.
    
    Args:
        img: полное изображение.
        by: номер блока по вертикали.
        bx: номер блока по горизонтали.
        bit: встраиваемый бит ЦВЗ.
        delta: шаг квантования.
    """
    cdef double A[NN_SVD]
    cdef double u[N_SVD]
    cdef double v[N_SVD]
    cdef double s1, s_new, delta_sigma

    cdef int i, j
    cdef int r0 = by * N_SVD
    cdef int c0 = bx * N_SVD

    for i in range(N_SVD):
        for j in range(N_SVD):
            A[i * N_SVD + j] = img[r0 + i, c0 + j]

    power_iteration(A, u, v, &s1)

    if s1 < 1e-12:
        return

    s_new = qim_embed(s1, bit, delta)

    delta_sigma = s_new - s1
    for i in range(N_SVD):
        for j in range(N_SVD):
            img[r0 + i, c0 + j] += delta_sigma * u[i] * v[j]

cdef int extract_block(double[:, ::1] img, int by, int bx,
                        double delta) noexcept nogil:
    """
    Извлечение одного бита ЦВЗ из блока 4x4.

    Args:
        img: полное изображение.
        by: номер блока по вертикали
        bx: номер блока по горизонтали
        delta: шаг квантования.

    Returns:
        извлечённый бит ЦВЗ.
    """
    cdef double A[NN_SVD]
    cdef int i, j
    cdef int r0 = by * N_SVD
    cdef int c0 = bx * N_SVD

    for i in range(N_SVD):
        for j in range(N_SVD):
            A[i * N_SVD + j] = img[r0 + i, c0 + j]

    return qim_extract(power_iteration_sigma(A), delta)


cdef double H5[5]
cdef bint FILTERS_INITIALIZED = False

cdef void init_filters() noexcept nogil:
    """
    Однократная инициализация C-массива коэффициентов фильтра.
    """
    global FILTERS_INITIALIZED
    if FILTERS_INITIALIZED:
        return
    H5[0] = 0.05
    H5[1] = 0.25
    H5[2] = 0.40
    H5[3] = 0.25
    H5[4] = 0.05
    FILTERS_INITIALIZED = True

cdef inline int reflect(int i, int n) noexcept nogil:
    """
    Зеркальное отражение индекса, вышедшего за границы массива.

    Args:
        i: индекс, возможно выходящий за границы.
        n: размер массива по данной оси.

    Returns:
        индекс в диапазоне [0, n).
    """
    if n == 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i
        if i >= n:
            i = 2 * (n - 1) - i
    return i

cdef void lowpass2d(double[:, :] src, double[:, :] dst, double[:, :] tmp,
                     double gain) noexcept nogil:
    """
    Двумерная низкочастотная фильтрация ядром Бёрта-Адельсона.

    Args:
        src: входная матрица.
        dst: выходная матрица.
        tmp: буфер для промежуточного результата после горизонтального прохода.
        gain: общий множитель результата.
    """
    cdef int H = src.shape[0]
    cdef int W = src.shape[1]
    cdef int i, j, t, jj, ii
    cdef double acc

    for i in range(H):
        for j in range(W):
            acc = 0.0
            for t in range(5):
                jj = reflect(j + t - 2, W)
                acc += H5[t] * src[i, jj]
            tmp[i, j] = acc

    for i in range(H):
        for j in range(W):
            acc = 0.0
            for t in range(5):
                ii = reflect(i + t - 2, H)
                acc += H5[t] * tmp[ii, j]
            dst[i, j] = gain * acc

cdef inline void downsample2(double[:, :] src, double[:, :] dst) noexcept nogil:
    """
    Прореживание изображения вдвое по обеим осям.

    Args:
        src: входная матрица.
        dst: выходная матрица вдвое меньшего размера.
    """
    cdef int Hc = dst.shape[0]
    cdef int Wc = dst.shape[1]
    cdef int i, j
    for i in range(Hc):
        for j in range(Wc):
            dst[i, j] = src[2 * i, 2 * j]

cdef inline void upsample2(double[:, :] src, double[:, :] dst) noexcept nogil:
    """
    Растяжение изображения вдвое по обеим осям с вставкой нулей.

    Args:
        src: входная матрица.
        dst: выходная матрица вдвое большего размера.
    """
    cdef int Hc = src.shape[0]
    cdef int Wc = src.shape[1]
    cdef int i, j
    for i in range(Hc):
        for j in range(Wc):
            dst[2 * i, 2 * j] = src[i, j]

cdef lp_analysis_step(double[:, :] x):
    """
    Один уровень анализа лапласовой пирамиды.

    Args:
        x: входное изображение данного уровня пирамиды, H и W должны быть чётными.

    Returns:
        coarse: грубое приближение, вход для следующего уровня пирамиды.
        band: полосовой остаток, вход для направленного банка фильтров.
    """
    cdef int H = x.shape[0]
    cdef int W = x.shape[1]
    cdef int Hc = H // 2
    cdef int Wc = W // 2
    cdef int i, j

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] lp = np.empty((H, W), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] tmp = np.empty((H, W), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] up = np.zeros((H, W), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] pred = np.empty((H, W), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] coarse = np.empty((Hc, Wc), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] band = np.empty((H, W), dtype=np.float64)

    cdef double[:, :] v_lp = lp
    cdef double[:, :] v_tmp = tmp
    cdef double[:, :] v_up = up
    cdef double[:, :] v_pred = pred
    cdef double[:, :] v_coarse = coarse
    cdef double[:, :] v_band = band

    with nogil:
        lowpass2d(x, v_lp, v_tmp, 1.0)
        downsample2(v_lp, v_coarse)

        upsample2(v_coarse, v_up)
        lowpass2d(v_up, v_pred, v_tmp, 4.0)

        for i in range(H):
            for j in range(W):
                v_band[i, j] = x[i, j] - v_pred[i, j]

    return coarse, band

cdef lp_synthesis_step(double[:, :] coarse, double[:, :] band):
    """
    Один уровень синтеза лапласовой пирамиды.

    Args:
        coarse: грубое приближение данного уровня.
        band: полосовой остаток данного уровня.

    Returns:
        out: восстановленное изображение уровнем выше.
    """
    cdef int H = band.shape[0]
    cdef int W = band.shape[1]
    cdef int i, j

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] up = np.zeros((H, W), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] tmp = np.empty((H, W), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] out = np.empty((H, W), dtype=np.float64)

    cdef double[:, :] v_up = up
    cdef double[:, :] v_tmp = tmp
    cdef double[:, :] v_out = out

    with nogil:
        upsample2(coarse, v_up)
        lowpass2d(v_up, v_out, v_tmp, 4.0)
        for i in range(H):
            for j in range(W):
                v_out[i, j] += band[i, j]

    return out

cdef double A_PRED = 10.0 / 32.0
cdef double B_PRED = -1.0 / 32.0
cdef double A_UPD = 5.0 / 32.0
cdef double B_UPD = -0.5 / 32.0

cdef int PRED_OFF[12][2]
cdef int UPD_OFF[12][2]
cdef bint OFF_INITIALIZED = False

cdef void init_offsets() noexcept nogil:
    """
    Однократное заполнение таблиц смещений лифтинг-шаблонов.
    """
    global OFF_INITIALIZED
    if OFF_INITIALIZED:
        return

    PRED_OFF[0][0] = 0;  PRED_OFF[0][1] = 0
    PRED_OFF[1][0] = 1;  PRED_OFF[1][1] = -1
    PRED_OFF[2][0] = 0;  PRED_OFF[2][1] = -1
    PRED_OFF[3][0] = 1;  PRED_OFF[3][1] = 0
    PRED_OFF[4][0] = -1; PRED_OFF[4][1] = 0
    PRED_OFF[5][0] = 1;  PRED_OFF[5][1] = -2
    PRED_OFF[6][0] = 0;  PRED_OFF[6][1] = 1
    PRED_OFF[7][0] = 2;  PRED_OFF[7][1] = -1
    PRED_OFF[8][0] = -1; PRED_OFF[8][1] = -1
    PRED_OFF[9][0] = 0;  PRED_OFF[9][1] = -2
    PRED_OFF[10][0] = 1; PRED_OFF[10][1] = 1
    PRED_OFF[11][0] = 2; PRED_OFF[11][1] = 0

    UPD_OFF[0][0] = 0;   UPD_OFF[0][1] = 0
    UPD_OFF[1][0] = -1;  UPD_OFF[1][1] = 1
    UPD_OFF[2][0] = 0;   UPD_OFF[2][1] = 1
    UPD_OFF[3][0] = -1;  UPD_OFF[3][1] = 0
    UPD_OFF[4][0] = -2;  UPD_OFF[4][1] = 1
    UPD_OFF[5][0] = 0;   UPD_OFF[5][1] = -1
    UPD_OFF[6][0] = -1;  UPD_OFF[6][1] = 2
    UPD_OFF[7][0] = 1;   UPD_OFF[7][1] = 0
    UPD_OFF[8][0] = -2;  UPD_OFF[8][1] = 0
    UPD_OFF[9][0] = -1;  UPD_OFF[9][1] = -1
    UPD_OFF[10][0] = 0;  UPD_OFF[10][1] = 2
    UPD_OFF[11][0] = 1;  UPD_OFF[11][1] = 1

    OFF_INITIALIZED = True

cdef inline int wrap(int v, int n) noexcept nogil:
    """
    Быстрое циклическое приведение индекса одним сложением или вычитанием.

    Args:
        v: индекс, возможно вышедший за границы, но находящийся в рамках -n <= v < 2n.
        n: период приведения (размер массива по данной оси).

    Returns:
        индекс в диапазоне [0, n).
    """
    if v < 0:
        return v + n
    if v >= n:
        return v - n
    return v

cdef inline int cmod(int v, int n) noexcept nogil:
    """
    Полное приведение по модулю с гарантированно неотрицательным
    результатом, для произвольного выхода индекса за границы.

    Args:
        v: индекс с произвольным выходом за границы.
        n: период приведения (размер массива по данной оси).

    Returns:
        индекс в диапазоне [0, n).
    """
    v = v % n
    if v < 0:
        v += n
    return v

cdef void lift_forward(double[:, :] a, double[:, :] b) noexcept nogil:
    """
    Прямой лифтинг: шаг предсказания, затем шаг обновления.

    Args:
        a: косет A (p + q чётно).
        b: косет B (p + q нечётно).
    """
    cdef int M = a.shape[0]
    cdef int Nh = a.shape[1]
    cdef int p, q, t, pp, qq
    cdef double s_near, s_far

    for p in range(M):
        for q in range(Nh):
            s_near = 0.0
            s_far = 0.0
            for t in range(4):
                pp = wrap(p + PRED_OFF[t][0], M)
                qq = wrap(q + PRED_OFF[t][1], Nh)
                s_near += a[pp, qq]
            for t in range(4, 12):
                pp = wrap(p + PRED_OFF[t][0], M)
                qq = wrap(q + PRED_OFF[t][1], Nh)
                s_far += a[pp, qq]
            b[p, q] -= A_PRED * s_near + B_PRED * s_far

    for p in range(M):
        for q in range(Nh):
            s_near = 0.0
            s_far = 0.0
            for t in range(4):
                pp = wrap(p + UPD_OFF[t][0], M)
                qq = wrap(q + UPD_OFF[t][1], Nh)
                s_near += b[pp, qq]
            for t in range(4, 12):
                pp = wrap(p + UPD_OFF[t][0], M)
                qq = wrap(q + UPD_OFF[t][1], Nh)
                s_far += b[pp, qq]
            a[p, q] += A_UPD * s_near + B_UPD * s_far

cdef void lift_inverse(double[:, :] a, double[:, :] b) noexcept nogil:
    """
    Обратный лифтинг: обратный шаг обновления, затем обратный шаг предсказания.

    Args:
        a: направленный низкочастотный канал, будущий косет A (p + q чётно).
        b: направленный высокочастотный канал, будущий косет B (p + q нечётно).
    """
    cdef int M = a.shape[0]
    cdef int Nh = a.shape[1]
    cdef int p, q, t, pp, qq
    cdef double s_near, s_far

    for p in range(M):
        for q in range(Nh):
            s_near = 0.0
            s_far = 0.0
            for t in range(4):
                pp = wrap(p + UPD_OFF[t][0], M)
                qq = wrap(q + UPD_OFF[t][1], Nh)
                s_near += b[pp, qq]
            for t in range(4, 12):
                pp = wrap(p + UPD_OFF[t][0], M)
                qq = wrap(q + UPD_OFF[t][1], Nh)
                s_far += b[pp, qq]
            a[p, q] -= A_UPD * s_near + B_UPD * s_far

    for p in range(M):
        for q in range(Nh):
            s_near = 0.0
            s_far = 0.0
            for t in range(4):
                pp = wrap(p + PRED_OFF[t][0], M)
                qq = wrap(q + PRED_OFF[t][1], Nh)
                s_near += a[pp, qq]
            for t in range(4, 12):
                pp = wrap(p + PRED_OFF[t][0], M)
                qq = wrap(q + PRED_OFF[t][1], Nh)
                s_far += a[pp, qq]
            b[p, q] += A_PRED * s_near + B_PRED * s_far

cdef void qx_split(double[:, :] x, double[:, :] a, double[:, :] b,
                    int modulate) noexcept nogil:
    """
    Анализ одной ступени квинканс-веерного банка фильтров.

    Args:
        x: входной массив.
        a: выходной направленный низкочастотный канал.
        b: выходной направленный высокочастотный канал.
        modulate:
            1 - применить модуляцию (-1)^i, дающую веерные фильтры.
            0 - без модуляции, ромбовидные фильтры.
    """
    cdef int M = x.shape[0]
    cdef int N = x.shape[1]
    cdef int Nh = N // 2
    cdef int p, q, i, j0, j1
    cdef double sgn

    for p in range(M):
        for q in range(Nh):
            i = wrap(p + q, M)
            j0 = cmod(p - q, N)
            j1 = j0 + 1 if j0 + 1 < N else 0
            sgn = -1.0 if (modulate != 0 and ((p + q) & 1) == 1) else 1.0
            a[p, q] = sgn * x[i, j0]
            b[p, q] = sgn * x[i, j1]

    lift_forward(a, b)

cdef void qx_merge(double[:, :] a, double[:, :] b, double[:, :] x,
                    int modulate) noexcept nogil:
    """
    Синтез одной ступени квинканс-веерного банка фильтров.

    Args:
        a: направленный низкочастотный канал.
        b: направленный высокочастотный канал.
        x: выходной массив полной ширины.
        modulate (int): снятие модуляции.
            Значение должно совпадать с использованным в qx_split.
    """
    cdef int M = x.shape[0]
    cdef int N = x.shape[1]
    cdef int Nh = N // 2
    cdef int p, q, i, j0, j1
    cdef double sgn

    lift_inverse(a, b)

    for p in range(M):
        for q in range(Nh):
            i = wrap(p + q, M)
            j0 = cmod(p - q, N)
            j1 = j0 + 1 if j0 + 1 < N else 0
            sgn = -1.0 if (modulate != 0 and ((p + q) & 1) == 1) else 1.0
            x[i, j0] = sgn * a[p, q]
            x[i, j1] = sgn * b[p, q]

def dfb_analysis(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] band, int levels):
    """
    Направленное разложение полосового остатка на 2^levels клиновидных
    поддиапазонов.

    Args:
        band: полосовой остаток Лапласовой пирамиды или произвольный массив.
        levels: число уровней дерева.

    Returns:
        список из 2^levels массивов
    """
    if levels <= 0:
        return [band]

    cdef int M = band.shape[0]
    cdef int N = band.shape[1]

    if N % 2 != 0:
        raise ValueError(f"Width {N} is not divided by 2: DFB cannot be continued.")
    if M % N != 0:
        raise ValueError(
            f"It is required for length M to be divided by width N."
            f"Both M and N need to be divided by 2^(n_levels + dfb_levels).")

    init_offsets()

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] a = np.empty((M, N // 2), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] b = np.empty((M, N // 2), dtype=np.float64)
    cdef double[:, :] v_x = band
    cdef double[:, :] v_a = a
    cdef double[:, :] v_b = b

    with nogil:
        qx_split(v_x, v_a, v_b, 1)

    return dfb_analysis(a, levels - 1) + dfb_analysis(b, levels - 1)

def dfb_synthesis(list subbands, int levels):
    """
    Обратное направленное разложение: сборка 2^levels клиновидных
    поддиапазонов обратно в полосовой остаток полной ширины.

    Args:
        subbands: список из 2^levels массивов
        levels (int): число уровней дерева.
            Должно совпадать со значением, использованным в dfb_analysis.

    Returns:
        out: восстановленный полосовой остаток.
    """
    if levels <= 0:
        return np.ascontiguousarray(subbands[0], dtype=np.float64)

    init_offsets()

    cdef int half = len(subbands) // 2
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] a = np.ascontiguousarray(
        dfb_synthesis(subbands[:half], levels - 1), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] b = np.ascontiguousarray(
        dfb_synthesis(subbands[half:], levels - 1), dtype=np.float64)

    cdef int M = a.shape[0]
    cdef int N = a.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] out = np.empty((M, 2 * N), dtype=np.float64)

    cdef double[:, :] v_a = a
    cdef double[:, :] v_b = b
    cdef double[:, :] v_out = out

    with nogil:
        qx_merge(v_a, v_b, v_out, 1)

    return out

cpdef contourlet_decompose(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] image,
                         int n_levels, int dfb_levels):
    """
    Прямое контурлет-преобразование = лапласова пирамида + DFB на каждом
    полосовом уровне.

    Args:
        image: входное изображение.
        n_levels: число уровней лапласовой пирамиды.
        dfb_levels: число уровней направленного дерева.

    Returns:
        (lowpass, bands) - кортеж:
            lowpass: грубое приближение, остаток пирамиды после всех уровней.
            bands: список списков, каждый из которых содержит
                2^dfb_levels направленных поддиапазонов масштаба k.
    """
    init_filters()

    cdef int H = image.shape[0]
    cdef int W = image.shape[1]
    cdef int k
    cdef int need = 1 << n_levels

    if n_levels < 1:
        raise ValueError("n_levels must be >= 1.")
    if H % need != 0 or W % need != 0:
        raise ValueError(
            f"Size {W}x{H} is not divided by 2^n_levels = {need}.")

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] cur = image
    bands = []

    for k in range(n_levels):
        coarse, band = lp_analysis_step(cur)
        bands.append(dfb_analysis(band, dfb_levels))
        cur = coarse

    return cur, bands

cpdef contourlet_reconstruct(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] lowpass,
                           list bands, int dfb_levels):
    """
    Обратное контурлет-преобразование: сборка изображения из грубого
    приближения и направленных поддиапазонов всех масштабов.

    Args:
        lowpass: грубое приближение; первый элемент кортежа из contourlet_decompose.
        bands (list): список списков, каждый из которых содержит
            2^dfb_levels направленных поддиапазонов масштаба k.
            Порядок элементов внутри каждого списка обязан совпадать
            с тем, что вернул dfb_analysis.
        dfb_levels: число уровней направленного дерева.
            Должно совпадать со значением, использованным в contourlet_decompose.

    Returns:
        cur: восстановленное изображение
    """
    init_filters()

    cdef int k
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] cur = lowpass
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] band

    for k in range(len(bands) - 1, -1, -1):
        band = np.ascontiguousarray(dfb_synthesis(bands[k], dfb_levels),
                                    dtype=np.float64)
        cur = np.ascontiguousarray(lp_synthesis_step(cur, band), dtype=np.float64)

    return cur