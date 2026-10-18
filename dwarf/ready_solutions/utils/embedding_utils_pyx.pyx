import operator as dwt_operator
from libc.math cimport cos, sqrt, fabs, floor, ceil, log10
from cpython.exc cimport PyErr_CheckSignals
import numpy as np
cimport numpy as cnp
cimport cython
cnp.import_array()


cdef double _C_DCT[8][8]
cdef bint _C_DCT_INITIALIZED = False


cdef cnp.ndarray validate_ycbcr_image(object image):
    """
    Проверяет структуру YCbCr-массива без преобразования его каналов.

    Args:
        image: массив float64 формы (H, W, 3) с конечными значениями Y, Cb, Cr;
            высота и ширина должны быть не меньше 8 для обработки блоками DCT.

    Returns:
        np.ndarray: непрерывный C-массив с исходными значениями каналов.
    """
    if image is None:
        raise ValueError("input_image is required")
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"input_image must have shape (H, W, 3) in Y, Cb, Cr order, got {arr.shape}")
    if arr.dtype != np.float64:
        raise TypeError(f"input_image must be YCbCr with dtype float64, got {arr.dtype}")
    if arr.shape[0] < 8 or arr.shape[1] < 8:
        raise ValueError("input_image must contain at least one complete 8x8 block")
    if not np.all(np.isfinite(arr)):
        raise ValueError("input_image must contain only finite YCbCr values")
    return np.ascontiguousarray(arr)


cdef void init_dct_matrix():
    """
    Инициализирует ортонормированную матрицу DCT размером 8x8.
    """
    global _C_DCT_INITIALIZED
    cdef int k, n
    cdef double scale
    cdef double pi = 3.1415926535897932384626433832795

    if _C_DCT_INITIALIZED:
        return
    for k in range(8):
        scale = sqrt(1.0 / 8.0) if k == 0 else sqrt(2.0 / 8.0)
        for n in range(8):
            _C_DCT[k][n] = scale * cos(pi * (2.0 * n + 1.0) * k / 16.0)
    _C_DCT_INITIALIZED = True


cdef void apply_dct_8x8(const double block_img[8][8], double block_dct[8][8]) noexcept nogil:
    """
    Выполняет двумерное DCT блока 8x8.

    Args:
        block_img: входной пространственный блок double размером 8x8.
        block_dct: выходной буфер double размером 8x8; матрица DCT должна быть заранее инициализирована.
    """
    cdef DBlock temp
    cdef int i, j, k
    cdef double value

    for i in range(8):
        for j in range(8):
            value = 0.0
            for k in range(8):
                value += _C_DCT[i][k] * block_img[k][j]
            temp[i][j] = value
    for i in range(8):
        for j in range(8):
            value = 0.0
            for k in range(8):
                value += temp[i][k] * _C_DCT[j][k]
            block_dct[i][j] = value


cdef void apply_idct_8x8(const double block_dct[8][8], double block_img[8][8]) noexcept nogil:
    """
    Выполняет обратное двумерное DCT блока 8x8.

    Args:
        block_dct: входной блок коэффициентов double размером 8x8.
        block_img: выходной буфер double размером 8x8; матрица DCT должна быть заранее инициализирована.
    """
    cdef DBlock temp
    cdef int i, j, k
    cdef double value

    for i in range(8):
        for j in range(8):
            value = 0.0
            for k in range(8):
                value += _C_DCT[k][i] * block_dct[k][j]
            temp[i][j] = value
    for i in range(8):
        for j in range(8):
            value = 0.0
            for k in range(8):
                value += temp[i][k] * _C_DCT[k][j]
            block_img[i][j] = value


cdef void apply_haar_dwt(double[:, ::1] coeffs, double[:, ::1] scratch, int levels) noexcept nogil:
    """
    Выполняет ортонормированное Haar DWT с рекурсией только по LL.

    Args:
        coeffs: изменяемая матрица; размеры делятся на 2**levels.
        scratch: отдельный рабочий буфер той же формы, не пересекающийся с coeffs.
        levels: число уровней; вызывающий код проверяет размеры и конечность.

    Квадранты: LL сверху слева, LH сверху справа, HL снизу слева, HH снизу справа.
    LH = (a-b+c-d)/2, HL = (a+b-c-d)/2; дополнения границ нет.
    """
    cdef Py_ssize_t height = coeffs.shape[0]
    cdef Py_ssize_t width = coeffs.shape[1]
    cdef Py_ssize_t half_h, half_w, r, c
    cdef int level
    cdef double a, b, x, d

    for level in range(levels):
        half_h, half_w = height // 2, width // 2
        for r in range(half_h):
            for c in range(half_w):
                a = coeffs[2 * r, 2 * c]
                b = coeffs[2 * r, 2 * c + 1]
                x = coeffs[2 * r + 1, 2 * c]
                d = coeffs[2 * r + 1, 2 * c + 1]
                scratch[r, c] = (a + b + x + d) / 2.0
                scratch[r, c + half_w] = (a - b + x - d) / 2.0
                scratch[r + half_h, c] = (a + b - x - d) / 2.0
                scratch[r + half_h, c + half_w] = (a - b - x + d) / 2.0
        for r in range(height):
            for c in range(width):
                coeffs[r, c] = scratch[r, c]
        height, width = half_h, half_w


cdef void apply_haar_idwt(double[:, ::1] coeffs, double[:, ::1] scratch, int levels) noexcept nogil:
    """
    Восстанавливает пространственную матрицу после apply_haar_dwt.

    Args:
        coeffs: изменяемая матрица коэффициентов с расположением LL, LH, HL, HH.
        scratch: отдельный рабочий буфер той же формы, не пересекающийся с coeffs.
        levels: число уровней прямого преобразования; размеры делятся на 2**levels.
    """
    cdef Py_ssize_t height = coeffs.shape[0] >> (levels - 1)
    cdef Py_ssize_t width = coeffs.shape[1] >> (levels - 1)
    cdef Py_ssize_t half_h, half_w, r, c
    cdef int level
    cdef double ll, lh, hl, hh

    for level in range(levels):
        half_h, half_w = height // 2, width // 2
        for r in range(half_h):
            for c in range(half_w):
                ll = coeffs[r, c]
                lh = coeffs[r, c + half_w]
                hl = coeffs[r + half_h, c]
                hh = coeffs[r + half_h, c + half_w]
                scratch[2 * r, 2 * c] = (ll + lh + hl + hh) / 2.0
                scratch[2 * r, 2 * c + 1] = (ll - lh + hl - hh) / 2.0
                scratch[2 * r + 1, 2 * c] = (ll + lh - hl - hh) / 2.0
                scratch[2 * r + 1, 2 * c + 1] = (ll - lh - hl + hh) / 2.0
        for r in range(height):
            for c in range(width):
                coeffs[r, c] = scratch[r, c]
        height *= 2
        width *= 2


cdef cnp.ndarray dwt_validate_image(object image, bint embedding):
    """
    Проверяет яркостную матрицу dwt и подготавливает непрерывный массив float64.

    Args:
        image: матрица Y формы (512, 512) с конечными значениями uint8 или float64.
        embedding: включает проверку диапазона пикселей 0…255 для встраивания.

    Returns:
        np.ndarray: непрерывная яркостная матрица float64 без преобразования цвета.
    """
    if image is None:
        raise ValueError("input_image is required")
    arr = np.asarray(image)
    if arr.shape != (512, 512):
        raise ValueError("input_image must be a grayscale Y plane of shape (512, 512)")
    if arr.dtype not in (np.dtype(np.float64), np.dtype(np.uint8)):
        raise TypeError("input_image must have dtype float64 or uint8")
    if not np.all(np.isfinite(arr)):
        raise ValueError("input_image must contain only finite values")
    if embedding and (np.any(arr < 0) or np.any(arr > 255)):
        raise ValueError("embedding input must lie in [0, 255]")
    return np.ascontiguousarray(arr, dtype=np.float64)


cdef cnp.ndarray dwt_validate_bits(object watermark):
    """
    Проверяет ровно 256 битов без неявного преобразования dtype.

    Args:
        watermark: одномерный массив uint8 с битами 0 и 1.

    Returns:
        np.ndarray: непрерывный массив 256 битов uint8.
    """
    arr = np.asarray(watermark)
    if arr.ndim != 1 or arr.size != 256:
        raise ValueError("watermark_bits must be a one-dimensional array of exactly 256 bits")
    if arr.dtype != np.uint8:
        raise TypeError("watermark_bits must have dtype uint8")
    if np.any((arr != 0) & (arr != 1)):
        raise ValueError("watermark_bits must contain only 0 and 1")
    return np.ascontiguousarray(arr)


cdef object dwt_key_material(object key):
    """
    Создаёт три независимых ключевых потока без глобального состояния RNG.

    Args:
        key: целое число от 0 до 2**128 - 1.

    Returns:
        tuple: перестановка 4096 позиций, опорные векторы формы (256, 32)
            и доли смещения длины 256; генераторы PCG64 созданы через SeedSequence.
    """
    key = dwt_integer("key", key, 0, (1 << 128) - 1)
    s_perm, s_u, s_d = np.random.SeedSequence(key).spawn(3)
    permutation = np.random.Generator(np.random.PCG64(s_perm)).permutation(4096)
    reference = np.random.Generator(np.random.PCG64(s_u)).standard_normal((256, 32))
    eta = 0.5 * np.random.Generator(np.random.PCG64(s_d)).random(256) - 0.25
    return np.ascontiguousarray(permutation, dtype=np.intp), reference, eta


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dwt_prepare_core(const double[:, ::1] coeffs, const cnp.intp_t[::1] permutation,
                        const double[:, ::1] reference, double[:, ::1] unit,
                        double[:, ::1] tangent, double[:, ::1] stats,
                        cnp.uint8_t[::1] flags) noexcept nogil:
    """
    Готовит исходные векторы до модификации любого поддиапазона.

    Args:
        coeffs: исходная матрица Haar float64 формы (512, 512), три уровня.
        permutation: ключевая перестановка 4096 позиций.
        reference: ключевые опорные векторы формы (256, 32).
        unit: выходные единичные опорные векторы формы (256, 32).
        tangent: выходные ортогональные направления формы (256, 32).
        stats: выходные корреляции, нормы и проекции формы (256, 4).
        flags: выходные битовые маски вырожденных случаев длины 256.
    """
    cdef dwt_vector x
    cdef int j, k, index, r, c, offset
    for j in range(256):
        offset = (j % 128) * 32
        for k in range(32):
            index = <int>permutation[offset + k]
            r, c = index // 64, index % 64
            if j < 128:
                x[k] = coeffs[r + 64, c + 64] - coeffs[r, c + 64]
            else:
                x[k] = coeffs[r + 64, c] - coeffs[r + 64, c + 64]
        flags[j] = dwt_prepare_vector(x, &reference[j, 0], &unit[j, 0],
                                        &tangent[j, 0], &stats[j, 0])


cdef object dwt_render(cnp.ndarray coefficients, const double[:, ::1] targets):
    """
    Синтезирует вещественное и округлённое изображения из исходного спектра.

    Args:
        coefficients: исходная матрица Haar float64 формы (512, 512), три уровня.
        targets: две целевые матрицы разностей формы (2, 4096) без перестановки.

    Returns:
        tuple: два новых массива float64; второй округлён половинами от нуля
            и ограничен диапазоном 0…255 для сохранения в uint8.
    """
    cdef cnp.ndarray result = coefficients.copy()
    cdef cnp.ndarray scratch_array = np.empty((512, 512), dtype=np.float64)
    cdef cnp.ndarray pixels_array = np.empty((512, 512), dtype=np.float64)
    cdef double[:, ::1] result_view = result
    cdef double[:, ::1] scratch = scratch_array
    cdef double[:, ::1] pixels = pixels_array
    with nogil:
        dwt_merge_bands(result_view, targets)
        apply_haar_idwt(result_view, scratch, 3)
        dwt_round_pixels(result_view, pixels)
    return result, pixels_array


cdef object dwt_calibrate(cnp.ndarray image, cnp.ndarray coefficients,
                        const double[:, ::1] unit, const double[:, ::1] tangent,
                        const double[:, ::1] stats, const cnp.intp_t[::1] permutation,
                        const double[::1] eta, const cnp.uint8_t[::1] bits,
                        double[:, ::1] targets, dwt_embed_core_callback embed_core):
    """
    Выбирает h полным перебором по отклонению PSNR от 42 дБ после uint8.

    Args:
        image: исходная целочисленная матрица Y float64 формы (512, 512).
        coefficients: исходная матрица Haar float64 формы (512, 512).
        unit: неизменные единичные опорные векторы формы (256, 32).
        tangent: неизменные ортогональные направления формы (256, 32).
        stats: неизменные корреляции, нормы и проекции формы (256, 4).
        permutation: ключевая перестановка 4096 позиций.
        eta: ключевые доли смещения длины 256.
        bits: проверенные 256 битов uint8.
        targets: рабочий выходной буфер целевых разностей формы (2, 4096).
        embed_core: Cython-ядро встраивания, вызываемое без GIL.

    Returns:
        tuple: выбранный h, SSE, число допустимых кандидатов и полных синтезов.
            Перебираются h=1…65536, при равном отклонении выбирается меньший h.
            У границ округления проверяется полный синтез; допуск PSNR 0,01 дБ.
            BER не вычисляется и не участвует в выборе.
    """
    if np.any(image != np.floor(image)):
        raise ValueError("automatic PSNR calibration requires integer-valued input pixels")
    cdef cnp.ndarray costs_array = np.empty((4096, 4, 511), dtype=np.uint32)
    cdef cnp.uint32_t[:, :, ::1] costs = costs_array
    cdef const double[:, ::1] source = image
    cdef const double[:, ::1] coeffs = coefficients
    cdef int h, best_h = 0, valid_count = 0, full_count = 0
    cdef long long sse, best_sse = 0
    cdef double psnr, error, best_error = float("inf")
    cdef bint valid, near_half
    with nogil:
        dwt_rounding_cost(source, costs)
    for h in range(1, 65537):
        if h % 1024 == 0:
            PyErr_CheckSignals()
        with nogil:
            valid = embed_core(unit, tangent, stats, permutation, eta, bits, h, targets)
            if valid:
                sse = dwt_rounding_sse(coeffs, targets, costs, &near_half)
        if not valid:
            continue
        valid_count += 1
        if near_half:
            _, pixels = dwt_render(coefficients, targets)
            sse = int(np.sum((pixels - image) ** 2))
            full_count += 1
        if sse == 0:
            continue
        psnr = 10.0 * log10(65025.0 * 262144.0 / sse)
        error = fabs(psnr - 42.0)
        if error < best_error:
            best_error, best_h, best_sse = error, h, sse
    if best_h == 0 or best_error > 0.01:
        raise RuntimeError(
            f"dwt target 42 +/- 0.01 dB not reached: best h={best_h}, "
            f"PSNR error={best_error} dB; no calibrated container returned"
        )
    return best_h, best_sse, valid_count, full_count


cdef object dwt_integer(object name, object value, object lower, object upper):
    """
    Проверяет целочисленный параметр dwt и его диапазон.

    Args:
        name: имя параметра для сообщения об ошибке.
        value: значение с поддержкой целочисленного индекса; bool не допускается.
        lower: включительная нижняя граница.
        upper: включительная верхняя граница.

    Returns:
        result: проверенное целое число в диапазоне [lower, upper].
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        value = dwt_operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}], got {value}")
    return value


cdef double dwt_quantize(double value, double delta, double dither, int bit) noexcept nogil:
    """Выбирает ближайшее допустимое слово dwt; половины округляет от нуля.

    Args:
        value: нормированное скалярное произведение в диапазоне [-1, 1].
        delta: проверенный общий шаг, 0 < delta <= 1.
        dither: ключевое смещение в [-delta/4, delta/4).
        bit: номер кодовой книги, 0 или 1.
    """
    cdef double offset = dither + bit * delta / 2.0
    cdef double scaled = (value - dither) / delta - bit / 2.0
    cdef double k = floor(scaled + 0.5) if scaled >= 0 else -floor(-scaled + 0.5)
    cdef double lower = ceil((-1.0 - offset) / delta)
    cdef double upper = floor((1.0 - offset) / delta)
    cdef double result
    if k < lower:
        k = lower
    elif k > upper:
        k = upper
    result = k * delta + offset
    # Границы выбираются среди слов решётки, а не обрезанием result до +/-1.
    if result < -1.0:
        result += delta
    elif result > 1.0:
        result -= delta
    return result


cdef int dwt_prepare_vector(const double x[32], const double u[32],
                             double unit[32], double tangent[32], double stats[4]) noexcept nogil:
    """Готовит ортонормальный базис и проекции исходного вектора dwt.

    Args:
        x: исходные 32 коэффициента матрицы разностей.
        u: ключевые 32 нормальные случайные величины.
        unit: выходной единичный опорный вектор; при нулевом u используется e0.
        tangent: выходное единичное ортогональное направление.
        stats: выход [feature, norm_x, dot(x, unit), dot(x, tangent)].

    Returns:
        int: битовая маска: 1 — нулевой x, 2 — параллельный x, 4 — нулевой u.
    """
    cdef int j, axis = 0, flags = 0
    cdef double nx = 0.0, nu = 0.0, feature = 0.0, nv = 0.0
    cdef double pu = 0.0, pv = 0.0, value
    for j in range(32):
        nx += x[j] * x[j]
        nu += u[j] * u[j]
    nx, nu = sqrt(nx), sqrt(nu)
    for j in range(32):
        unit[j] = u[j] / nu if nu > 1e-12 else (1.0 if j == 0 else 0.0)
    if nu <= 1e-12:
        flags |= 4
    if nx > 1e-12:
        for j in range(32):
            feature += (x[j] / nx) * unit[j]
        feature = min(1.0, max(-1.0, feature))
        for j in range(32):
            tangent[j] = x[j] / nx - feature * unit[j]
            nv += tangent[j] * tangent[j]
        nv = sqrt(nv)
    else:
        flags |= 1
    if nv <= 1e-12:
        if nx > 1e-12:
            flags |= 2
        for j in range(1, 32):
            if fabs(unit[j]) < fabs(unit[axis]):
                axis = j
        nv = 0.0
        for j in range(32):
            value = (1.0 if j == axis else 0.0) - unit[axis] * unit[j]
            tangent[j] = value
            nv += value * value
        nv = sqrt(nv)
    for j in range(32):
        tangent[j] /= nv
        pu += x[j] * unit[j]
        pv += x[j] * tangent[j]
    stats[0], stats[1], stats[2], stats[3] = feature, nx, pu, pv
    return flags


cdef bint dwt_embed_vector(const double unit[32], const double tangent[32],
                            const double stats[4], int bit, double delta,
                            double dither, double output[32]) noexcept nogil:
    """Встраивает бит формулами (7)-(15), сохраняя проекционную норму.

    Args:
        unit, tangent, stats: базис и проекции из dwt_prepare_vector.
        bit: целевой бит 0 или 1.
        delta, dither: шаг и ключевое смещение квантователя.
        output: выходной вектор из 32 изменённых коэффициентов.

    Returns:
        bool: False при неположительной проекции ненулевого исходного вектора.
    """
    cdef double q = dwt_quantize(stats[0], delta, dither, bit)
    cdef double sine = sqrt(max(0.0, 1.0 - q * q))
    cdef double norm = 0.0, rho, factor
    cdef int j
    for j in range(32):
        output[j] = q * unit[j] + sine * tangent[j]
        norm += output[j] * output[j]
    norm = sqrt(norm)
    if stats[1] <= 1e-12:
        rho = sqrt(32.0)
    else:
        rho = (q * stats[2] + sine * stats[3]) / norm
        if rho <= 1e-12:
            return False
    factor = rho / norm
    for j in range(32):
        output[j] *= factor
    return True


cdef int dwt_decode_vector(const double x[32], const double u[32], double delta,
                             double dither, int* erased, int* ambiguous,
                             double* margin) noexcept nogil:
    """Декодирует ближайшую книгу без исходника; стирания дают бит 0.

    Args:
        x: полученный вектор из 32 коэффициентов.
        u: воспроизведённый ключевой вектор.
        delta, dither: согласованные шаг и смещение.
        erased, ambiguous: выходные флаги нулевой нормы и равных расстояний.
        margin: выходная абсолютная разность расстояний до двух книг.
    """
    cdef double nx = 0.0, nu = 0.0, feature = 0.0, e0, e1
    cdef int j
    erased[0], ambiguous[0], margin[0] = 0, 0, 0.0
    for j in range(32):
        nx += x[j] * x[j]
        nu += u[j] * u[j]
    nx, nu = sqrt(nx), sqrt(nu)
    if nx <= 1e-12:
        erased[0] = 1
        return 0
    if nu <= 1e-12:
        feature = x[0] / nx
    else:
        for j in range(32):
            feature += (x[j] / nx) * (u[j] / nu)
    feature = min(1.0, max(-1.0, feature))
    e0 = fabs(feature - dwt_quantize(feature, delta, dither, 0))
    e1 = fabs(feature - dwt_quantize(feature, delta, dither, 1))
    ambiguous[0] = e0 == e1
    margin[0] = fabs(e0 - e1)
    return 0 if e0 <= e1 else 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dwt_merge_bands(double[:, ::1] coeffs, const double[:, ::1] targets) noexcept nogil:
    """Минимизирует MSE трёх поддиапазонов при двух заданных разностях.

    Args:
        coeffs: изменяемые исходные коэффициенты Haar 512x512, три уровня.
        targets: две построчные матрицы разностей формы (2, 4096), уже без перестановки.
    """
    cdef int r, c, i
    cdef double a, b, h, d1, d2
    for r in range(64):
        for c in range(64):
            i = 64 * r + c
            a, b, h = coeffs[r, c + 64], coeffs[r + 64, c], coeffs[r + 64, c + 64]
            d1, d2 = targets[0, i] - (h - a), targets[1, i] - (b - h)
            coeffs[r, c + 64] = a - (2.0 * d1 + d2) / 3.0
            coeffs[r + 64, c] = b + (d1 + 2.0 * d2) / 3.0
            coeffs[r + 64, c + 64] = h + (d1 - d2) / 3.0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dwt_round_pixels(const double[:, ::1] source, double[:, ::1] output) noexcept nogil:
    """Округляет половины от нуля и ограничивает пиксели до преобразования uint8.

    Args:
        source: вещественное изображение после обратной DWT.
        output: отдельный выходной буфер той же формы, целочисленные значения float64.
    """
    cdef Py_ssize_t r, c
    cdef double x
    for r in range(source.shape[0]):
        for c in range(source.shape[1]):
            x = source[r, c]
            x = floor(x + 0.5) if x >= 0 else -floor(-x + 0.5)
            output[r, c] = min(255.0, max(0.0, x))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dwt_rounding_cost(const double[:, ::1] source,
                             cnp.uint32_t[:, :, ::1] costs) noexcept nogil:
    """Кэширует точную SSE целочисленных сдвигов каждого квадранта 4x4.

    Args:
        source: проверенное целочисленное изображение 512x512 в диапазоне 0..255.
        costs: выход (4096, 4, 511); последний индекс соответствует сдвигу -255..255.

    Сдвиги за этим диапазоном дают ту же сатурацию. Кэш ускоряет полный
    перебор h; изменение коэффициентов уровня 3 постоянно на квадрантах 4x4.
    """
    cdef int hist[256]
    cdef int cell, quadrant, r0, c0, r, c, n, pos_count, neg_count
    cdef unsigned int positive, negative
    for cell in range(4096):
        for quadrant in range(4):
            for n in range(256):
                hist[n] = 0
            r0 = (cell // 64) * 8 + (quadrant // 2) * 4
            c0 = (cell % 64) * 8 + (quadrant % 2) * 4
            for r in range(r0, r0 + 4):
                for c in range(c0, c0 + 4):
                    hist[<int>source[r, c]] += 1
            positive, negative = 0, 0
            pos_count, neg_count = 16 - hist[255], 16 - hist[0]
            costs[cell, quadrant, 255] = 0
            for n in range(1, 256):
                positive += (2 * n - 1) * pos_count
                negative += (2 * n - 1) * neg_count
                costs[cell, quadrant, 255 + n] = positive
                costs[cell, quadrant, 255 - n] = negative
                pos_count -= hist[255 - n]
                neg_count -= hist[n]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long long dwt_rounding_sse(const double[:, ::1] coeffs,
                                 const double[:, ::1] targets,
                                 const cnp.uint32_t[:, :, ::1] costs,
                                 bint* near_half) noexcept nogil:
    """Оценивает SSE после uint8 без повторного синтеза 512x512 для каждого h.

    Args:
        coeffs: исходные коэффициенты Haar 512x512.
        targets: две требуемые матрицы разностей формы (2, 4096).
        costs: кэш dwt_rounding_cost.
        near_half: флаг необходимости полного синтеза у границы округления.

    Returns:
        int: сумма квадратов ошибок по всем пикселям после ограничения диапазона.
    """
    cdef int i, r, c, q, shift
    cdef double d1, d2, da, db, dh, change[4], rounded
    cdef long long total = 0
    near_half[0] = False
    for i in range(4096):
        r, c = i // 64, i % 64
        d1 = targets[0, i] - (coeffs[r + 64, c + 64] - coeffs[r, c + 64])
        d2 = targets[1, i] - (coeffs[r + 64, c] - coeffs[r + 64, c + 64])
        da, db, dh = -(2.0 * d1 + d2) / 3.0, (d1 + 2.0 * d2) / 3.0, (d1 - d2) / 3.0
        change[0] = (da + db + dh) / 8.0
        change[1] = (-da + db - dh) / 8.0
        change[2] = (da - db - dh) / 8.0
        change[3] = (-da - db + dh) / 8.0
        for q in range(4):
            rounded = floor(change[q] + 0.5)
            if fabs(change[q] + 0.5 - rounded) < 1e-9:
                near_half[0] = True
            shift = <int>min(255.0, max(-255.0, rounded))
            total += costs[i, q, shift + 255]
    return total


import math as contourlet_math
import operator as contourlet_operator
from scipy.ndimage import affine_transform as contourlet_affine_transform
from scipy.ndimage import convolve1d as contourlet_convolve1d


CONTOURLET_WU_ANALYSIS_TAPS = np.array(
    [
        -0.010182337649086284,
        0.0,
        0.019233304448274095,
        0.0,
        -0.037193816690412405,
        0.0,
        0.06873077913133242,
        0.0,
        -0.1364716087690037,
        0.0,
        0.44547727214752497,
        0.7071067811865476,
        0.44547727214752497,
        0.0,
        -0.1364716087690037,
        0.0,
        0.06873077913133242,
        0.0,
        -0.037193816690412405,
        0.0,
        0.019233304448274095,
        0.0,
        -0.010182337649086284,
    ],
    dtype=np.float64,
)


CONTOURLET_WU_SYNTHESIS_TAPS = np.array(
    [
        -0.0001466256621468425,
        0.0,
        0.0005539191681102939,
        0.0,
        -0.0015943278016769325,
        0.0,
        0.004002790066940809,
        0.0,
        -0.009625731475207483,
        0.0,
        0.027484278919498686,
        -0.010182337649086284,
        -0.032441663141041334,
        0.019233304448274095,
        0.04522994383724128,
        -0.037193816690412405,
        -0.05667209121196076,
        0.06873077913133242,
        0.06618649579553824,
        -0.1364716087690037,
        -0.07235863289861688,
        0.44547727214752497,
        0.7816205624171361,
        0.44547727214752497,
        -0.07235863289861688,
        -0.1364716087690037,
        0.06618649579553824,
        0.06873077913133242,
        -0.05667209121196076,
        -0.037193816690412405,
        0.04522994383724128,
        0.019233304448274095,
        -0.032441663141041334,
        -0.010182337649086284,
        0.027484278919498686,
        0.0,
        -0.009625731475207483,
        0.0,
        0.004002790066940809,
        0.0,
        -0.0015943278016769325,
        0.0,
        0.0005539191681102939,
        0.0,
        -0.0001466256621468425,
    ],
    dtype=np.float64,
)


CONTOURLET_WU_SWAP_XY = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void contourlet_luma_core(const double[:, :, :] rgb, double[:, ::1] result) noexcept nogil:
    """
    Вычисляет яркость в одном проходе без промежуточных RGB-плоскостей.

    Args:
        rgb: проверенный RGB-массив float64 с произвольными шагами.
        result: выходная яркостная матрица формы (H, W).
    """
    cdef Py_ssize_t row, column
    cdef double red, green, blue
    for row in range(rgb.shape[0]):
        for column in range(rgb.shape[1]):
            red = 0.299 * rgb[row, column, 0]
            green = 0.587 * rgb[row, column, 1]
            blue = 0.114 * rgb[row, column, 2]
            result[row, column] = (red + green) + blue


cdef object contourlet_validate_image(object value):
    """
    Проверяет входное RGB-изображение uint8 без изменения данных.

    Args:
        value: массив формы (H, W, 3).

    Returns:
        Проверенный массив изображения.
    """
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"input_image must have shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"input_image must have dtype uint8, got {image.dtype}")
    if image.shape[0] < 16 or image.shape[1] < 16:
        raise ValueError("input_image is too small for contourlet watermarking")
    return image


cdef object contourlet_validate_bits(object value):
    """
    Проверяет одномерный двоичный ЦВЗ uint8.

    Args:
        value: последовательность битов ЦВЗ.

    Returns:
        Непрерывный массив битов uint8.
    """
    bits = np.asarray(value)
    if bits.ndim != 1:
        raise ValueError("watermark_bits must be a one-dimensional array")
    if bits.dtype != np.uint8:
        raise TypeError(f"watermark_bits must have dtype uint8, got {bits.dtype}")
    if bits.size == 0:
        raise ValueError("watermark_bits must not be empty")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("watermark_bits must contain only 0 and 1")
    return np.ascontiguousarray(bits)


cdef double contourlet_positive_float(object name, object value):
    """
    Проверяет положительный конечный числовой параметр.

    Args:
        name: имя параметра.
        value: проверяемое значение.

    Returns:
        Проверенное значение float.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


cdef object contourlet_integer(object name, object value, object minimum=0):
    """
    Проверяет целочисленный параметр и его нижнюю границу.

    Args:
        name: имя параметра.
        value: проверяемое значение.
        minimum: минимально допустимое значение.

    Returns:
        Проверенное целое число.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = contourlet_operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {result}")
    return result


cdef bint contourlet_boolean(object name, object value):
    """
    Проверяет логический параметр.

    Args:
        name: имя параметра.
        value: проверяемое значение.

    Returns:
        Логическое значение параметра.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


cdef object contourlet_choice(object name, object value, object choices):
    """
    Проверяет строковый параметр по набору допустимых значений.

    Args:
        name: имя параметра.
        value: проверяемая строка.
        choices: допустимые значения.

    Returns:
        Проверенная строка.
    """
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {allowed}")
    return value


cdef object contourlet_to_luma(object image):
    """
    Вычисляет яркость RGB с коэффициентами 0.299, 0.587 и 0.114.

    Args:
        image: RGB-массив формы (H, W, 3).

    Returns:
        Яркостная матрица float64.
    """
    rgb = image.astype(np.float64, copy=False)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    result = np.empty((rgb.shape[0], rgb.shape[1]), dtype=np.float64)
    cdef const double[:, :, :] source_view = rgb
    cdef double[:, ::1] target_view = result
    with nogil:
        contourlet_luma_core(source_view, target_view)
    return result


cdef object contourlet_lowpass_analysis(object image, object coefficient_scale=1.0):
    """
    Вычисляет низкочастотную ветвь одной ступени PKVA.

    Args:
        image: двумерное изображение.
        coefficient_scale: масштаб коэффициентов.

    Returns:
        Непрерывная матрица LL float64.
    """
    filtered = contourlet_convolve1d(np.asarray(image, dtype=np.float64), CONTOURLET_WU_ANALYSIS_TAPS, axis=0, mode="wrap")
    filtered = contourlet_convolve1d(filtered, CONTOURLET_WU_ANALYSIS_TAPS, axis=1, mode="wrap")
    return np.ascontiguousarray(filtered[::2, ::2] * coefficient_scale)


cdef object contourlet_lowpass_synthesis_delta(object coarse_delta, object coefficient_scale=1.0):
    """
    Переносит поправку LL в пространственную область.

    Args:
        coarse_delta: изменение коэффициентов LL.
        coefficient_scale: масштаб коэффициентов.

    Returns:
        Пространственная поправка float64.
    """
    coarse_delta = np.asarray(coarse_delta, dtype=np.float64)
    expanded = np.zeros((2 * coarse_delta.shape[0], 2 * coarse_delta.shape[1]), dtype=np.float64)
    expanded[::2, ::2] = coarse_delta / coefficient_scale
    expanded = contourlet_convolve1d(expanded, CONTOURLET_WU_SYNTHESIS_TAPS, axis=0, mode="wrap")
    return contourlet_convolve1d(expanded, CONTOURLET_WU_SYNTHESIS_TAPS, axis=1, mode="wrap")


cdef object contourlet_blocks(object image, object block_size):
    """
    Представляет изображение неперекрывающимися квадратными блоками.

    Args:
        image: двумерный массив.
        block_size: сторона блока.

    Returns:
        Массив блоков с двумя пространственными осями.
    """
    rows = image.shape[0] // block_size
    columns = image.shape[1] // block_size
    active = image[: rows * block_size, : columns * block_size]
    return active.reshape(rows, block_size, columns, block_size).transpose(0, 2, 1, 3)


cdef object contourlet_adaptive_steps(object blocks, object delta0):
    """
    Вычисляет шаги квантования по формуле исходной реализации.

    Args:
        blocks: массив блоков коэффициентов.
        delta0: базовый шаг квантования.

    Returns:
        Шаг для каждого блока с ограничением энергии снизу единицей.
    """
    energy = np.mean(np.square(blocks, dtype=np.float64), axis=(-2, -1))


    energy = np.maximum(energy, 1.0)
    return np.log2((energy + 1000.0) / 1000.0) * delta0


cdef object contourlet_embed_core(
    object lowpass,
    object bits,
    object delta0,
    object block_size,
    object max_iterations=6,
):
    """
    Встраивает биты в наибольшие сингулярные значения блоков LL.

    Args:
        lowpass: матрица LL.
        bits: двоичный ЦВЗ uint8.
        delta0: базовый шаг.
        block_size: сторона блока.
        max_iterations: предел повторных проходов квантования.

    Returns:
        Изменённая матрица LL float64.
    """
    result = np.array(lowpass, dtype=np.float64, copy=True)
    carriers = contourlet_blocks(result, block_size)
    flat = np.ascontiguousarray(carriers).reshape(-1, block_size, block_size)
    if bits.size > flat.shape[0]:
        raise ValueError(f"Not enough payload capacity: need {bits.size} bits, available {flat.shape[0]}")

    selected = flat[: bits.size]


    for iteration in range(max_iterations):
        steps = contourlet_adaptive_steps(selected, delta0)
        u, singular, vh = np.linalg.svd(selected, full_matrices=False)
        rounded = np.floor(singular[:, 0] / steps + 0.5)
        parity = rounded.astype(np.int64) % 2
        decoded = 1 - np.ceil(singular[:, 0] / steps).astype(np.int64) % 2
        if iteration and np.array_equal(decoded, bits):
            break
        singular[:, 0] = np.where(
            parity == bits,
            rounded * steps + steps / 2.0,
            rounded * steps - steps / 2.0,
        )
        selected[:] = (u * singular[:, None, :]) @ vh
    carriers[:] = flat.reshape(carriers.shape)
    return result


cdef object contourlet_extract_core(object lowpass, object num_bits, object delta0, object block_size):
    """
    Извлекает биты по чётности квантованных сингулярных значений.

    Args:
        lowpass: матрица LL.
        num_bits: число извлекаемых битов.
        delta0: базовый шаг.
        block_size: сторона блока.

    Returns:
        Непрерывный массив извлечённых битов int8.
    """
    carriers = contourlet_blocks(np.asarray(lowpass, dtype=np.float64), block_size)
    flat = carriers.reshape(-1, block_size, block_size)
    if num_bits > flat.shape[0]:
        raise ValueError(f"Not enough payload capacity: need {num_bits} bits, available {flat.shape[0]}")
    selected = flat[:num_bits]
    steps = contourlet_adaptive_steps(selected, delta0)
    singular = np.linalg.svd(selected, compute_uv=False)
    quantized = np.ceil(singular[:, 0] / steps).astype(np.int64)
    return np.ascontiguousarray((1 - quantized % 2).astype(np.int8))


cdef object contourlet_arnold(
    object bits,
    object iterations,
    object inverse=False,
    object matrix=(1, 1, 1, 2),
):
    """
    Выполняет прямую или обратную перестановку Arnold.

    Args:
        bits: одномерный массив битов квадратного ЦВЗ.
        iterations: число перестановок; ноль отключает обработку.
        inverse: флаг обратного преобразования.
        matrix: четыре коэффициента модульной матрицы.

    Returns:
        Переставленный одномерный массив битов.
    """
    if iterations == 0:
        return np.ascontiguousarray(bits)
    side = int(round(bits.size**0.5))
    if side * side != bits.size:
        raise ValueError("Arnold scrambling requires a square binary watermark")

    source = np.asarray(bits).reshape(side, side)
    yy, xx = np.indices((side, side))
    a, b, c, d = matrix
    determinant = a * d - b * c
    if contourlet_math.gcd(determinant, side) != 1:
        raise ValueError("Arnold matrix determinant must be invertible modulo the watermark side")
    if inverse:
        determinant_inverse = pow(determinant, -1, side)
        a, b, c, d = (
            determinant_inverse * d,
            -determinant_inverse * b,
            -determinant_inverse * c,
            determinant_inverse * a,
        )
    target_x = (a * xx + b * yy) % side
    target_y = (c * xx + d * yy) % side

    result = source
    for _ in range(iterations):
        transformed = np.empty_like(result)
        transformed[target_y, target_x] = result[yy, xx]
        result = transformed
    return np.ascontiguousarray(result.reshape(-1))


cdef double contourlet_weighted_moment(object weights, object x, object y, object p, object q):
    """
    Вычисляет момент с нормировкой на суммарную интенсивность.

    Args:
        weights: веса интенсивности.
        x: координаты по горизонтали.
        y: координаты по вертикали.
        p: степень x.
        q: степень y.

    Returns:
        Взвешенный момент float.
    """
    return float(np.sum(weights * x**p * y**q) / np.sum(weights))


cdef object contourlet_moment_transform(
    object image,
    object target_std,
    object output_shape,
    object sign_mode='fifth_moment',
):
    """
    Строит моментное аффинное преобразование исходной реализации.

    Args:
        image: яркостная матрица.
        target_std: масштаб нормализованных координат.
        output_shape: форма выходного растра.
        sign_mode: правило выбора знаков масштабирования.

    Returns:
        Матрица прямого преобразования 3x3.
    """
    weights = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
    yy, xx = np.indices(weights.shape, dtype=np.float64)
    mass = weights.sum()
    if mass <= np.finfo(np.float64).eps:
        raise ValueError("image has zero intensity mass and cannot be normalized")

    centroid = np.array([(weights * xx).sum() / mass, (weights * yy).sum() / mass])
    dx = xx - centroid[0]
    dy = yy - centroid[1]
    mu02 = contourlet_weighted_moment(weights, dx, dy, 0, 2)
    if mu02 <= 1e-12:
        raise ValueError("image moments are degenerate and cannot define an affine normalization")


    beta = -contourlet_weighted_moment(weights, dx, dy, 1, 1) / mu02
    shear = np.array([[1.0, beta], [0.0, 1.0]], dtype=np.float64)
    sheared_x = dx + beta * dy
    sheared_y = dy


    mu20_sheared = contourlet_weighted_moment(weights, sheared_x, sheared_y, 2, 0)
    mu02_sheared = contourlet_weighted_moment(weights, sheared_x, sheared_y, 0, 2)
    if min(mu20_sheared, mu02_sheared) <= 1e-12:
        raise ValueError("image moments are degenerate and cannot define an affine normalization")
    if sign_mode == "fifth_moment":
        mu50_sheared = contourlet_weighted_moment(weights, sheared_x, sheared_y, 5, 0)
        mu05_sheared = contourlet_weighted_moment(weights, sheared_x, sheared_y, 0, 5)
        alpha_sign = 1.0 if mu50_sheared >= 0.0 else -1.0
        delta_sign = 1.0 if mu05_sheared >= 0.0 else -1.0
    elif sign_mode == "positive":


        alpha_sign = delta_sign = 1.0
    else:
        raise ValueError(f"unsupported normalization sign mode: {sign_mode!r}")
    alpha = alpha_sign / np.sqrt(mu20_sheared)
    delta = delta_sign / np.sqrt(mu02_sheared)
    scaling = np.diag([alpha, delta])
    scaled = scaling @ shear
    scaled_x = scaled[0, 0] * dx + scaled[0, 1] * dy
    scaled_y = scaled[1, 0] * dx + scaled[1, 1] * dy


    numerator = contourlet_weighted_moment(weights, scaled_x, scaled_y, 3, 0) + contourlet_weighted_moment(
        weights, scaled_x, scaled_y, 1, 2
    )
    denominator = contourlet_weighted_moment(weights, scaled_x, scaled_y, 2, 1) + contourlet_weighted_moment(
        weights, scaled_x, scaled_y, 0, 3
    )
    if abs(numerator) <= 1e-15 and abs(denominator) <= 1e-15:
        angle = 0.0
    else:
        ratio = np.copysign(np.inf, numerator) if abs(denominator) <= 1e-15 else numerator / denominator
        angle = -np.arctan(ratio)
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    oriented = rotation @ scaled

    output_center = np.array([(output_shape[1] - 1.0) / 2.0, (output_shape[0] - 1.0) / 2.0])
    matrix = target_std * oriented
    transform = np.eye(3, dtype=np.float64)
    transform[:2, :2] = matrix
    transform[:2, 2] = output_center - matrix @ centroid
    return transform


cdef object contourlet_warp_forward(object image, object source_to_normalized, object output_shape):
    """
    Переносит изображение в нормализованную координатную сетку.

    Args:
        image: яркостная матрица.
        source_to_normalized: матрица прямого преобразования 3x3.
        output_shape: форма результата.

    Returns:
        Результат билинейной интерполяции float64.
    """
    inverse = np.linalg.inv(source_to_normalized)
    matrix = CONTOURLET_WU_SWAP_XY @ inverse[:2, :2] @ CONTOURLET_WU_SWAP_XY
    offset = CONTOURLET_WU_SWAP_XY @ inverse[:2, 2]
    return contourlet_affine_transform(
        np.asarray(image, dtype=np.float64),
        matrix,
        offset=offset,
        output_shape=output_shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


cdef object contourlet_warp_inverse_delta(
    object normalized_delta,
    object source_to_normalized,
    object output_shape,
):
    """
    Возвращает нормализованную поправку в исходную сетку.

    Args:
        normalized_delta: нормализованная поправка.
        source_to_normalized: матрица прямого преобразования 3x3.
        output_shape: форма исходного изображения.

    Returns:
        Поправка в исходных координатах float64.
    """
    matrix = CONTOURLET_WU_SWAP_XY @ source_to_normalized[:2, :2] @ CONTOURLET_WU_SWAP_XY
    offset = CONTOURLET_WU_SWAP_XY @ source_to_normalized[:2, 2]
    return contourlet_affine_transform(
        np.asarray(normalized_delta, dtype=np.float64),
        matrix,
        offset=offset,
        output_shape=output_shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


cdef object contourlet_full_bounds_transform(object image_shape, object source_to_normalized):
    """
    Смещает преобразование к полному охватывающему растру.

    Args:
        image_shape: исходные высота и ширина.
        source_to_normalized: матрица преобразования 3x3.

    Returns:
        Скорректированная матрица и форма выходного растра.
    """
    height, width = image_shape
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [width - 1.0, 0.0, 1.0],
            [0.0, height - 1.0, 1.0],
            [width - 1.0, height - 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    mapped = (source_to_normalized @ corners.T).T[:, :2]
    lower = np.floor(mapped.min(axis=0))
    upper = np.ceil(mapped.max(axis=0))
    output_shape = (
        max(1, int(upper[1] - lower[1] + 1.0)),
        max(1, int(upper[0] - lower[0] + 1.0)),
    )
    adjusted = source_to_normalized.copy()
    adjusted[:2, 2] -= lower
    return adjusted, output_shape


cdef object contourlet_invariant_centroid(object image, object radius, object iterations=20):
    """
    Итерационно находит центроид интенсивности внутри круга.

    Args:
        image: яркостная матрица.
        radius: радиус круга.
        iterations: предел итераций.

    Returns:
        Координаты центра (x, y).
    """
    weights = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
    yy, xx = np.indices(weights.shape, dtype=np.float64)
    mass = weights.sum()
    if mass <= np.finfo(np.float64).eps:
        return (image.shape[1] - 1.0) / 2.0, (image.shape[0] - 1.0) / 2.0
    centroid = np.array([(weights * xx).sum() / mass, (weights * yy).sum() / mass])
    for _ in range(iterations):
        mask = (xx - centroid[0]) ** 2 + (yy - centroid[1]) ** 2 <= radius**2
        local = weights * mask
        local_mass = local.sum()
        if local_mass <= np.finfo(np.float64).eps:
            break
        updated = np.array([(local * xx).sum() / local_mass, (local * yy).sum() / local_mass])
        if np.linalg.norm(updated - centroid) < 0.01:
            centroid = updated
            break
        centroid = updated
    return float(centroid[0]), float(centroid[1])


cdef object contourlet_roi_origin(object image, object roi_size, object centroid_radius):
    """
    Размещает ROI вокруг инвариантного центроида внутри изображения.

    Args:
        image: яркостная матрица.
        roi_size: сторона ROI.
        centroid_radius: радиус поиска центроида.

    Returns:
        Координаты верхнего левого угла (row, column).
    """
    center_x, center_y = contourlet_invariant_centroid(image, centroid_radius)
    top = int(round(center_y - (roi_size - 1.0) / 2.0))
    left = int(round(center_x - (roi_size - 1.0) / 2.0))
    top = min(max(top, 0), image.shape[0] - roi_size)
    left = min(max(left, 0), image.shape[1] - roi_size)
    return top, left


cdef object contourlet_centered_roi_origin(object image, object roi_size):
    """
    Размещает ROI по геометрическому центру изображения.

    Args:
        image: двумерный массив.
        roi_size: сторона ROI.

    Returns:
        Координаты верхнего левого угла (row, column).
    """
    return (image.shape[0] - roi_size) // 2, (image.shape[1] - roi_size) // 2


cdef object contourlet_normalized_luma(
    object image,
    object normalize,
    object target_std,
    object canonical_shape,
    object bounds_mode='fixed',
    object sign_mode='fifth_moment',
):
    """
    Получает яркость и выполняет выбранную моментную нормализацию.

    Args:
        image: RGB-изображение.
        normalize: флаг нормализации.
        target_std: масштаб координат.
        canonical_shape: форма фиксированного растра.
        bounds_mode: политика границ растра.
        sign_mode: правило знаков масштабирования.

    Returns:
        Яркостная матрица и матрица преобразования 3x3.
    """
    luma = contourlet_to_luma(image)
    if not normalize:
        return luma.copy(), np.eye(3, dtype=np.float64)
    transform = contourlet_moment_transform(luma, target_std, canonical_shape, sign_mode=sign_mode)
    output_shape = canonical_shape
    if bounds_mode == "full":
        transform, output_shape = contourlet_full_bounds_transform(luma.shape, transform)
    elif bounds_mode != "fixed":
        raise ValueError(f"unsupported normalization bounds mode: {bounds_mode!r}")
    return contourlet_warp_forward(luma, transform, output_shape), transform


cdef object contourlet_profile_geometry(object roi_size, object block_size, object canonical_shape):
    """
    Проверяет кратность ROI и размеры канонического растра.

    Args:
        roi_size: сторона ROI.
        block_size: сторона блока.
        canonical_shape: форма канонического растра.
    """
    if roi_size % (2 * block_size):
        raise ValueError("roi_size must be divisible by 2 * block_size")
    if min(canonical_shape) < roi_size:
        raise ValueError(f"roi_size={roi_size} does not fit canonical_shape={canonical_shape}")
