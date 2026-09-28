import operator as dwt_operator
import hashlib as dwt_hashlib
import struct as dwt_struct
from libc.stdint cimport uint64_t
from libc.math cimport cos, sqrt, fabs, floor, ceil
import numpy as np
cimport numpy as cnp
cimport cython
cnp.import_array()
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    N_SVD, NN_SVD, MAX_SWEEPS, MAX_POWER_ITER
)


cdef double _C_DCT[8][8]
cdef bint _C_DCT_INITIALIZED = False


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


cdef cnp.ndarray validate_rgb_image(object image):
    """
    Проверяет RGB-изображение для обработки блоками 8x8.

    Args:
        image: изображение uint8 формы (H, W, 3), где H и W не меньше 8.

    Returns:
        rgb: C-contiguous массив uint8 исходной формы; создание копии не гарантируется.
    """
    if image is None:
        raise ValueError("input_image is required")
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"input_image must have shape (H, W, 3), got {arr.shape}")
    if arr.dtype != np.uint8:
        raise TypeError(f"input_image must have dtype uint8, got {arr.dtype}")
    if arr.shape[0] < 8 or arr.shape[1] < 8:
        raise ValueError("input_image must contain at least one complete 8x8 block")
    return np.ascontiguousarray(arr)


cdef cnp.ndarray rgb_to_ycbcr(const cnp.uint8_t[:, :, ::1] rgb):
    """
    Преобразует RGB в яркость и две цветоразностные компоненты.

    Args:
        rgb: входное RGB-изображение uint8 формы (H, W, 3).

    Returns:
        ycbcr: новый массив float64 формы (H, W, 3) с Y, Cb, Cr без смещений и округления.
    """
    cdef Py_ssize_t h = rgb.shape[0], w = rgb.shape[1], i, j
    cdef cnp.ndarray result = np.empty((h, w, 3), dtype=np.float64)
    cdef double[:, :, ::1] dst = result
    cdef double r, g, b
    for i in range(h):
        for j in range(w):
            r, g, b = rgb[i, j, 0], rgb[i, j, 1], rgb[i, j, 2]
            dst[i, j, 0] = 0.299 * r + 0.587 * g + 0.114 * b
            dst[i, j, 1] = -0.169 * r - 0.331 * g + 0.500 * b
            dst[i, j, 2] = 0.500 * r - 0.419 * g - 0.081 * b
    return result


cdef cnp.ndarray ycbcr_to_rgb(const double[:, :, ::1] ycbcr):
    """
    Преобразует YCbCr в RGB с округлением и ограничением диапазона.

    Args:
        ycbcr: массив double формы (H, W, 3) с компонентами Y, Cb, Cr без смещений.

    Returns:
        rgb: новый C-contiguous массив uint8 формы (H, W, 3); округление np.rint, диапазон 0..255.
    """
    cdef Py_ssize_t h = ycbcr.shape[0], w = ycbcr.shape[1], i, j
    cdef cnp.ndarray result = np.empty((h, w, 3), dtype=np.float64)
    cdef double[:, :, ::1] dst = result
    cdef double y, cb, cr
    for i in range(h):
        for j in range(w):
            y, cb, cr = ycbcr[i, j, 0], ycbcr[i, j, 1], ycbcr[i, j, 2]
            dst[i, j, 0] = y - 0.001 * cb + 1.402 * cr
            dst[i, j, 1] = y - 0.344 * cb - 0.714 * cr
            dst[i, j, 2] = y + 1.772 * cb + 0.001 * cr
    return np.ascontiguousarray(np.clip(np.rint(result), 0, 255), dtype=np.uint8)


DWT_PROFILE = "KH1998-P1"


_DWT_COMMON_DEFAULTS = {
    "levels": 4,
    "q": 4,
    "repetitions": 1,
    "key_seed": 100,
    "span_epsilon": 1e-9,
    "geometry_mode": "roi",
}


cdef object dwt_integer(object name, object value, object lower, object upper):
    """
    Проверяет целочисленный параметр DWT и его диапазон.

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


cdef object dwt_parameters(object parameters, bint extraction=False):
    """
    Дополняет и проверяет параметры DWT.

    Args:
        parameters: словарь параметров levels, q, repetitions, key_seed, span_epsilon и geometry_mode.
        extraction: разрешает extraction_mode со значениями all или coarsest при извлечении.

    Returns:
        config: новый словарь проверенных параметров с настройками по умолчанию; неизвестные ключи вызывают
            ошибку.
    """
    allowed = set(_DWT_COMMON_DEFAULTS)
    if extraction:
        allowed.add("extraction_mode")
    unknown = set(parameters) - allowed
    if unknown:
        raise TypeError("Unknown DWT parameter(s): " + ", ".join(sorted(unknown)))
    out = {**_DWT_COMMON_DEFAULTS, **parameters}
    out["levels"] = dwt_integer("levels", out["levels"], 1, 20)
    out["q"] = dwt_integer("q", out["q"], 1, 1048576)
    out["repetitions"] = dwt_integer("repetitions", out["repetitions"], 1, 2147483647)
    out["key_seed"] = dwt_integer("key_seed", out["key_seed"], 0, (1 << 64) - 1)
    eps = out["span_epsilon"]
    if isinstance(eps, (bool, np.bool_)) or not isinstance(
        eps, (int, float, np.integer, np.floating)
    ):
        raise TypeError("span_epsilon must be a real number")
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("span_epsilon must be finite and positive")
    out["span_epsilon"] = eps
    geometry_mode = out["geometry_mode"]
    if not isinstance(geometry_mode, str) or geometry_mode not in ("roi", "strict"):
        raise ValueError("geometry_mode must be 'roi' or 'strict'")
    if extraction:
        mode = out.setdefault("extraction_mode", "all")
        if not isinstance(mode, str) or mode not in ("all", "coarsest"):
            raise ValueError("extraction_mode must be 'all' or 'coarsest'")
    return out


cdef object dwt_shape(object height, object width, int levels):
    """
    Проверяет размеры рабочей области многоуровневого DWT.

    Args:
        height: положительная высота области в пикселях, не больше 2147483647.
        width: положительная ширина области в пикселях, не больше 2147483647.
        levels: заранее проверенное число уровней; размеры должны делиться на 2**levels.

    Returns:
        shape: кортеж (height, width); вход каждой ступени содержит не менее 10 отсчётов по каждой оси.
    """
    height = dwt_integer("height", height, 1, 2147483647)
    width = dwt_integer("width", width, 1, 2147483647)
    factor = 1 << levels
    if height % factor or width % factor:
        raise ValueError(f"Image dimensions must be divisible by 2**levels={factor}")
    if min(height, width) // (factor // 2) < 10:
        raise ValueError("Each DWT analysis input axis must have at least 10 samples")
    return height, width


cdef object dwt_rgb(object image):
    """
    Проверяет тип и форму RGB-изображения для DWT.

    Args:
        image: RGB-изображение uint8 формы (H, W, 3).

    Returns:
        rgb: C-contiguous массив uint8 исходной формы; создание копии не гарантируется.
    """
    if image is None:
        raise ValueError("input_image is required")
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"input_image must have shape (H, W, 3), got {arr.shape}")
    if arr.dtype != np.uint8:
        raise TypeError(f"input_image must have dtype uint8, got {arr.dtype}")
    return np.ascontiguousarray(arr)


cdef object dwt_bits(object watermark):
    """
    Проверяет бинарное сообщение для DWT.

    Args:
        watermark: непустой одномерный массив uint8 со значениями 0 и 1.

    Returns:
        bits: проверенный C-contiguous массив uint8; создание копии не гарантируется.
    """
    if watermark is None:
        raise ValueError("watermark_bits is required")
    arr = np.asarray(watermark)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("watermark_bits must be a non-empty one-dimensional array")
    if arr.dtype != np.uint8:
        raise TypeError("watermark_bits must have dtype uint8")
    if np.any((arr != 0) & (arr != 1)):
        raise ValueError("watermark_bits must contain only 0 and 1")
    return np.ascontiguousarray(arr)


cdef void dwt_check_capacity(object shape, object num_bits, object config) except *:
    """
    Проверяет, достаточно ли троек DWT для сообщения с повторениями.

    Args:
        shape: форма рабочей области; первые два элемента задают высоту и ширину.
        num_bits: заранее проверенная положительная длина сообщения в битах.
        config: проверенный словарь параметров с levels и repetitions.
    """
    height, width = dwt_shape(shape[0], shape[1], config["levels"])
    available = (height >> config["levels"]) * (width >> config["levels"])
    needed = num_bits * config["repetitions"]
    if needed > available:
        raise ValueError(
            f"Not enough DWT capacity: need {needed} triples per level "
            f"({num_bits} bits x {config['repetitions']} repetitions), "
            f"coarsest level has {available}; maximum bits="
            f"{available // config['repetitions']}"
        )


cpdef object dwt_geometry(object image_shape, object levels=4, object geometry_mode="roi"):
    """
    Определяет рабочую область DWT в левом верхнем углу изображения.

    Args:
        image_shape: форма (H, W) либо (H, W, 3) с положительными целыми размерами.
        levels: число уровней DWT от 1 до 20.
        geometry_mode: roi отбрасывает неполные нижнюю и правую полосы; strict требует кратности 2**levels.

    Returns:
        geometry: словарь размеров, координат ROI, неиспользуемых полос и признака transform_compatible;
            недостаточный размер отмечается False.
    """
    try:
        shape = tuple(image_shape)
    except TypeError as exc:
        raise TypeError("image_shape must contain (H,W) or (H,W,3)") from exc
    if len(shape) not in (2, 3):
        raise ValueError("image_shape must contain (H,W) or (H,W,3)")
    if len(shape) == 3:
        dwt_integer("channels", shape[2], 3, 3)
    levels = dwt_integer("levels", levels, 1, 20)
    height = dwt_integer("height", shape[0], 1, 2147483647)
    width = dwt_integer("width", shape[1], 1, 2147483647)
    if not isinstance(geometry_mode, str) or geometry_mode not in ("roi", "strict"):
        raise ValueError("geometry_mode must be 'roi' or 'strict'")
    factor = 1 << levels
    if geometry_mode == "strict" and (height % factor or width % factor):
        raise ValueError(f"Image dimensions must be divisible by 2**levels={factor} "
                         "in geometry_mode='strict'; use geometry_mode='roi'")
    effective_h = (height // factor) * factor
    effective_w = (width // factor) * factor
    minimum_axis = 5 * factor
    compatible = min(effective_h, effective_w) >= minimum_axis
    return {
        "geometry_mode": geometry_mode,
        "geometry_adapter": "top-left-roi-v1" if geometry_mode == "roi" else "strict-grid-v1",
        "image_shape": (height, width),
        "effective_shape": (effective_h, effective_w),
        "roi_yxyx": (0, 0, effective_h, effective_w),
        "unused_bottom_rows": height - effective_h,
        "unused_right_columns": width - effective_w,
        "grid_factor": factor,
        "minimum_axis": minimum_axis,
        "transform_compatible": compatible,
    }


cpdef object dwt_capacity(object image_shape, object levels=4,
                           object repetitions=1, object geometry_mode="roi"):
    """
    Вычисляет геометрическую ёмкость DWT в битах.

    Args:
        image_shape: форма изображения (H, W) либо (H, W, 3).
        levels: число уровней DWT от 1 до 20.
        repetitions: положительное число носителей на бит на каждом уровне.
        geometry_mode: режим выделения области roi либо проверки сетки strict.

    Returns:
        capacity_bits: максимальное число бит по размеру самого грубого уровня; 0 для слишком малой области, без
            проверки информативности носителей.
    """
    config = dwt_parameters({"levels": levels, "repetitions": repetitions,
                             "geometry_mode": geometry_mode})
    geometry = dwt_geometry(image_shape, config["levels"], config["geometry_mode"])
    if not geometry["transform_compatible"]:
        return 0
    height, width = geometry["effective_shape"]
    return ((height >> config["levels"]) * (width >> config["levels"])) // config["repetitions"]


cdef object dwt_image_region(object image, object config):
    """
    Выделяет допустимую рабочую RGB-область DWT.

    Args:
        image: проверенное RGB-изображение uint8 формы (H, W, 3).
        config: проверенные параметры DWT с levels и geometry_mode.

    Returns:
        result: пара (region, geometry), где region — C-contiguous RGB-массив рабочей области, geometry —
            словарь её геометрии.
    """
    geometry = dwt_geometry(image.shape, config["levels"], config["geometry_mode"])
    if not geometry["transform_compatible"]:
        raise ValueError(
            "Each DWT analysis input axis must have at least 10 samples: "
            f"levels={config['levels']} requires an aligned ROI with both axes >= "
            f"{geometry['minimum_axis']}; got {geometry['effective_shape']}. "
            "Use fewer levels or a larger image; no resize is performed automatically."
        )
    height, width = geometry["effective_shape"]
    return np.ascontiguousarray(image[:height, :width, :]), geometry


cdef object dwt_luminance(object rgb):
    """
    Вычисляет яркостную компоненту RGB-изображения.

    Args:
        rgb: RGB-массив uint8 формы (H, W, 3).

    Returns:
        luminance: C-contiguous массив float64 формы (H, W), равный 0.299*R + 0.587*G + 0.114*B.
    """
    return np.ascontiguousarray(
        0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2],
        dtype=np.float64,
    )


cdef object dwt_merge_region(object image, object luminance, object modified,
                              object geometry):
    """
    Переносит изменение яркости рабочей области в RGB-изображение.

    Args:
        image: исходное RGB-изображение uint8 формы (H, W, 3).
        luminance: исходная яркость рабочей области, двумерный массив float64.
        modified: изменённая яркость той же формы, двумерный массив float64.
        geometry: словарь геометрии с effective_shape для рабочей области.

    Returns:
        output: новый C-contiguous RGB-массив uint8; вне рабочей области пиксели сохранены, внутри применены
            np.rint и ограничение 0..255.
    """
    height, width = geometry["effective_shape"]
    delta = modified - luminance
    changed = image[:height, :width, :].astype(np.float64) + delta[:, :, None]
    region = np.ascontiguousarray(np.clip(np.rint(changed), 0, 255), dtype=np.uint8)
    output = image.copy(order="C")
    output[:height, :width, :] = region
    return output


cdef object dwt_require_decoded_bits(object details):
    """
    Проверяет наличие голосов для каждого извлечённого бита DWT.

    Args:
        details: словарь декодирования с массивами bits и missing_bits.

    Returns:
        bits: исходный массив bits из details; при наличии пропущенных битов возбуждается ValueError.
    """
    if np.any(details["missing_bits"]):
        indices = np.flatnonzero(details["missing_bits"])
        raise ValueError(
            f"DWT: {indices.size} bit(s) have no informative votes; "
            f"first indices={indices[:8].tolist()}. "
            "Use DWT.diagnostics for the vote/erasure arrays."
        )
    return details["bits"]


cdef object dwt_embed_coefficients_inplace(double[:, ::1] coeffs,
                        const unsigned char[::1] bits, int levels, int q,
                        Py_ssize_t repetitions, uint64_t key_seed,
                        double span_epsilon):
    """
    Встраивает сообщение в тройки коэффициентов на всех уровнях DWT.

    Args:
        coeffs: изменяемая упакованная матрица double формы (H, W) с заранее проверенными геометрией и ёмкостью.
        bits: непустой непрерывный массив unsigned char со значениями 0 и 1.
        levels: число уровней разложения.
        q: положительный параметр сетки из 2*q узлов между минимумом и максимумом тройки.
        repetitions: число повторений каждого бита на каждом уровне.
        key_seed: беззнаковый 64-битный ключ выбора троек.
        span_epsilon: положительный порог вырожденности размаха тройки.
    """
    cdef Py_ssize_t height = coeffs.shape[0], width = coeffs.shape[1]
    cdef Py_ssize_t hh, ww, needed = bits.shape[0] * repetitions
    cdef int level
    cdef object positions_array
    cdef const Py_ssize_t[::1] positions
    for level in range(1, levels + 1):
        hh, ww = height >> level, width >> level
        positions_array = dwt_positions(hh * ww, needed, key_seed, level, height, width)
        positions = positions_array
        dwt_embed_band(coeffs, hh, ww, positions, bits, q, span_epsilon, level)
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object dwt_decode_coefficients(const double[:, ::1] coeffs, Py_ssize_t num_bits,
                          int levels, int q, Py_ssize_t repetitions,
                          uint64_t key_seed, double span_epsilon,
                          bint coarsest_only):
    """
    Декодирует биты DWT по голосам выбранных троек коэффициентов.

    Args:
        coeffs: упакованная матрица double формы (H, W) с заранее проверенными геометрией и ёмкостью.
        num_bits: положительная длина извлекаемого сообщения.
        levels: число уровней разложения.
        q: параметр сетки квантования, совпадающий со встраиванием.
        repetitions: число повторений бита на уровне, совпадающее со встраиванием.
        key_seed: беззнаковый 64-битный ключ выбора троек и разрешения равенства голосов.
        span_epsilon: положительный порог, ниже или на котором тройка считается стиранием.
        coarsest_only: True для последнего уровня, False для всех уровней.

    Returns:
        details: словарь с bits (int8: 0, 1 или -1 без голосов), votes, erased_votes, tied_bits, missing_bits и
            параметрами голосования.
    """
    cdef Py_ssize_t height = coeffs.shape[0], width = coeffs.shape[1]
    cdef Py_ssize_t hh, ww, needed = num_bits * repetitions, i
    cdef int level, start_level = levels if coarsest_only else 1
    cdef object positions_array
    cdef const Py_ssize_t[::1] positions
    cdef object votes_array = np.zeros((num_bits, 2), dtype=np.longlong)
    cdef object erased_array = np.zeros(num_bits, dtype=np.longlong)
    cdef object output_array = np.empty(num_bits, dtype=np.int8)
    cdef object tie_array = dwt_tie_bits(num_bits, key_seed)
    cdef long long[:, ::1] votes = votes_array
    cdef long long[::1] erased = erased_array
    cdef signed char[::1] output = output_array
    cdef const unsigned char[::1] tie_bits = tie_array
    for level in range(start_level, levels + 1):
        hh, ww = height >> level, width >> level
        positions_array = dwt_positions(hh * ww, needed, key_seed, level, height, width)
        positions = positions_array
        with nogil:
            dwt_extract_band(coeffs, hh, ww, positions, num_bits, q,
                            span_epsilon, votes, erased)
    with nogil:
        for i in range(num_bits):
            if votes[i, 0] + votes[i, 1] == 0:
                output[i] = -1
            elif votes[i, 1] > votes[i, 0]:
                output[i] = 1
            elif votes[i, 0] > votes[i, 1]:
                output[i] = 0
            else:
                output[i] = <signed char>tie_bits[i]
    return {
        "profile": DWT_PROFILE,
        "bits": output_array,
        "votes": votes_array,
        "erased_votes": erased_array,
        "tied_bits": (votes_array[:, 0] == votes_array[:, 1]) & (output_array != -1),
        "missing_bits": output_array == -1,
        "extraction_mode": "coarsest" if coarsest_only else "all",
        "votes_per_bit": repetitions * (1 if coarsest_only else levels),
    }


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void dwt_filters(double* h, double* g) noexcept nogil:
    """
    Заполняет десятиотсчётные фильтры анализа DWT.

    Args:
        h: выходной буфер double длины 10 для низкочастотного фильтра корреляции.
        g: выходной буфер double длины 10 для знакопеременно отражённого высокочастотного фильтра.
    """
    h[0] = 0.1601023979741929
    h[1] = 0.6038292697971895
    h[2] = 0.7243085284377726
    h[3] = 0.1384281459013203
    h[4] = -0.2422948870663823
    h[5] = -0.0322448695846381
    h[6] = 0.0775714938400459
    h[7] = -0.0062414902127983
    h[8] = -0.0125807519990820
    h[9] = 0.0033357252854738
    cdef int t
    for t in range(10):
        g[t] = h[9 - t] if t % 2 == 0 else -h[9 - t]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _dwt_forward_level(double[:, ::1] work, double[:, ::1] temp,
                         Py_ssize_t height, Py_ssize_t width,
                         const double* h, const double* g) noexcept nogil:
    """
    Выполняет один уровень периодического двумерного DWT.

    Args:
        work: изменяемая матрица double, содержащая рабочую область в левом верхнем углу.
        temp: отдельный рабочий буфер double не меньше области (height, width).
        height: чётная высота области, не меньше 10.
        width: чётная ширина области, не меньше 10.
        h: низкочастотный фильтр double длины 10.
        g: высокочастотный фильтр double длины 10.
    """
    cdef Py_ssize_t r, c, k, idx, half_h = height // 2, half_w = width // 2
    cdef int t
    cdef double a, d, value
    for r in range(height):
        for k in range(half_w):
            a = 0.0
            d = 0.0
            for t in range(10):
                idx = (2 * k + t) % width
                value = work[r, idx]
                a += h[t] * value
                d += g[t] * value
            temp[r, k] = a
            temp[r, k + half_w] = d
    for c in range(width):
        for k in range(half_h):
            a = 0.0
            d = 0.0
            for t in range(10):
                idx = (2 * k + t) % height
                value = temp[idx, c]
                a += h[t] * value
                d += g[t] * value
            work[k, c] = a
            work[k + half_h, c] = d


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _dwt_inverse_level(double[:, ::1] work, double[:, ::1] temp,
                         Py_ssize_t height, Py_ssize_t width,
                         const double* h, const double* g) noexcept nogil:
    """
    Восстанавливает один уровень периодического двумерного DWT.

    Args:
        work: изменяемая матрица double с поддиапазонами [[LL, V], [H, D]] в рабочей области.
        temp: отдельный рабочий буфер double не меньше области (height, width).
        height: чётная высота восстанавливаемой области, не меньше 10.
        width: чётная ширина восстанавливаемой области, не меньше 10.
        h: низкочастотный фильтр double длины 10, использованный при анализе.
        g: высокочастотный фильтр double длины 10, использованный при анализе.
    """
    cdef Py_ssize_t r, c, k, idx, half_h = height // 2, half_w = width // 2
    cdef int t
    cdef double a, d

    for c in range(width):
        for r in range(height):
            temp[r, c] = 0.0
        for k in range(half_h):
            a = work[k, c]
            d = work[k + half_h, c]
            for t in range(10):
                idx = (2 * k + t) % height
                temp[idx, c] += h[t] * a + g[t] * d
    for r in range(height):
        for c in range(width):
            work[r, c] = 0.0
        for k in range(half_w):
            a = temp[r, k]
            d = temp[r, k + half_w]
            for t in range(10):
                idx = (2 * k + t) % width
                work[r, idx] += h[t] * a + g[t] * d


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef object dwt_forward(const double[:, ::1] source, int levels):
    """
    Выполняет многоуровневое двумерное DWT.

    Args:
        source: пространственная матрица double формы (H, W), заранее проверенная через dwt_shape.
        levels: число уровней разложения, использованное при проверке размеров.

    Returns:
        coeffs: новая C-contiguous матрица float64 исходной формы с вложенной упаковкой [[LL, V], [H, D]].
    """
    cdef object out = np.array(source, dtype=np.float64, order="C", copy=True)
    cdef object scratch = np.empty_like(out)
    cdef double[:, ::1] work = out
    cdef double[:, ::1] temp = scratch
    cdef double h[10]
    cdef double g[10]
    cdef Py_ssize_t height = source.shape[0], width = source.shape[1]
    cdef int level
    with nogil:
        dwt_filters(h, g)
        for level in range(levels):
            _dwt_forward_level(work, temp, height, width, h, g)
            height //= 2
            width //= 2
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef object dwt_inverse(const double[:, ::1] source, int levels):
    """
    Выполняет обратное многоуровневое двумерное DWT.

    Args:
        source: упакованная матрица коэффициентов double формы (H, W) с заранее проверенными размерами.
        levels: число уровней исходного разложения.

    Returns:
        image: новая C-contiguous пространственная матрица float64 формы (H, W); source не изменяется.
    """
    cdef object out = np.array(source, dtype=np.float64, order="C", copy=True)
    cdef object scratch = np.empty_like(out)
    cdef double[:, ::1] work = out
    cdef double[:, ::1] temp = scratch
    cdef double h[10]
    cdef double g[10]
    cdef Py_ssize_t height = source.shape[0], width = source.shape[1]
    cdef int level
    with nogil:
        dwt_filters(h, g)
        for level in range(levels - 1, -1, -1):
            _dwt_inverse_level(work, temp, height >> level, width >> level, h, g)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef uint64_t _dwt_next64(uint64_t* state) noexcept nogil:
    """
    Получает следующее псевдослучайное число SplitMix64.

    Args:
        state: указатель на изменяемое состояние uint64_t; арифметика выполняется по модулю 2**64.

    Returns:
        value: беззнаковое 64-битное число; состояние state обновлено без обращения к глобальному генератору.
    """
    cdef uint64_t z
    state[0] += <uint64_t>0x9E3779B97F4A7C15
    z = state[0]
    z = (z ^ (z >> 30)) * <uint64_t>0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * <uint64_t>0x94D049BB133111EB
    return z ^ (z >> 31)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef uint64_t _dwt_bounded(uint64_t* state, uint64_t bound) noexcept nogil:
    """
    Выбирает ограниченное псевдослучайное число методом отбраковки.

    Args:
        state: указатель на изменяемое состояние SplitMix64 типа uint64_t.
        bound: заранее проверенная положительная верхняя граница типа uint64_t.

    Returns:
        value: число uint64_t из [0, bound) без смещения от простого взятия остатка; state обновлено.
    """
    cdef uint64_t limit = ((<uint64_t>0xFFFFFFFFFFFFFFFF) // bound) * bound
    cdef uint64_t value
    while True:
        value = _dwt_next64(state)
        if value < limit:
            return value % bound


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef object dwt_positions(Py_ssize_t count, Py_ssize_t needed,
                         uint64_t key_seed, int level,
                         Py_ssize_t height, Py_ssize_t width):
    """
    Выбирает уникальные адреса троек DWT по ключу.

    Args:
        count: положительное общее число троек в поддиапазонах уровня.
        needed: число выбираемых троек от 0 до count; проверяется вызывающей стороной.
        key_seed: беззнаковый 64-битный ключ.
        level: номер уровня разложения, начиная с 1.
        height: высота полной матрицы коэффициентов.
        width: ширина полной матрицы коэффициентов.

    Returns:
        positions: отсортированный массив np.intp длины needed с плоскими адресами; выбор задаётся SHA-256 и
            SplitMix64.
    """
    cdef bytes payload = (b"DWARF-KH1998-P1/positions\0" +
                          dwt_struct.pack("<QQQI", key_seed, height, width, level))
    cdef uint64_t state = int.from_bytes(dwt_hashlib.sha256(payload).digest()[:8], "little")
    cdef object result = np.arange(count, dtype=np.intp)
    cdef Py_ssize_t[::1] positions = result
    cdef Py_ssize_t i, j, swap
    with nogil:
        for i in range(needed):
            j = i + <Py_ssize_t>_dwt_bounded(&state, <uint64_t>(count - i))
            swap = positions[i]
            positions[i] = positions[j]
            positions[j] = swap

    return np.sort(result[:needed]).copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef object dwt_tie_bits(Py_ssize_t length, uint64_t key_seed):
    """
    Формирует независимые биты для разрешения равенства голосов DWT.

    Args:
        length: неотрицательное число требуемых битов.
        key_seed: беззнаковый 64-битный ключ отдельного потока разрешения равенств.

    Returns:
        bits: непрерывный массив uint8 длины length со значениями 0 и 1, привязанными к индексам битов.
    """
    cdef bytes payload = b"DWARF-KH1998-P1/ties\0" + dwt_struct.pack("<Q", key_seed)
    cdef uint64_t state = int.from_bytes(dwt_hashlib.sha256(payload).digest()[:8], "little")
    cdef object result = np.empty(length, dtype=np.uint8)
    cdef unsigned char[::1] out = result
    cdef Py_ssize_t i
    with nogil:
        for i in range(length):
            out[i] = <unsigned char>(_dwt_next64(&state) >> 63)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void dwt_sort3(double* values, int* indices) noexcept nogil:
    """
    Стабильно сортирует тройку коэффициентов по возрастанию.

    Args:
        values: изменяемый буфер double длины 3 с коэффициентами в порядке H, V, D.
        indices: выходной буфер int длины 3 для исходных индексов отсортированных значений.
    """
    cdef double value
    cdef int index, a, b, step
    indices[0] = 0
    indices[1] = 1
    indices[2] = 2
    for step in range(3):
        if step == 1:
            a, b = 1, 2
        else:
            a, b = 0, 1
        if values[a] > values[b]:
            value = values[a]
            values[a] = values[b]
            values[b] = value
            index = indices[a]
            indices[a] = indices[b]
            indices[b] = index


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dwt_quantize(double a, double b, double c, int bit,
                        int q) noexcept nogil:
    """
    Квантует медиану тройки DWT для заданного бита.

    Args:
        a: минимальный коэффициент невырожденной тройки.
        b: медианный коэффициент; требуется a <= b <= c.
        c: максимальный коэффициент, строго больше a.
        bit: встраиваемый бит 0 или 1.
        q: положительный параметр равномерной сетки из 2*q узлов.

    Returns:
        median: ближайший узел double с чётностью индекса bit; при равных расстояниях выбран меньший индекс.
    """
    cdef int last = 2 * q - 1
    cdef double u = ((b - a) / (c - a)) * last
    cdef int j = <int>floor((u - bit) / 2.0)
    cdef int k, upper
    if j < 0:
        j = 0
    elif j >= q:
        j = q - 1
    k = 2 * j + bit
    upper = k + 2
    if upper <= last and fabs(u - upper) < fabs(u - k):
        k = upper

    if k == 0:
        return a
    if k == last:
        return c
    return a + (c - a) * (k / <double>last)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int dwt_decode(double a, double b, double c, int q) noexcept nogil:
    """
    Извлекает бит по чётности ближайшего узла сетки DWT.

    Args:
        a: минимальный коэффициент невырожденной тройки.
        b: медианный коэффициент; требуется a <= b <= c.
        c: максимальный коэффициент, строго больше a.
        q: положительный параметр сетки из 2*q узлов, использованный при встраивании.

    Returns:
        bit: целое 0 или 1; при равных расстояниях выбирается узел с меньшим индексом.
    """
    cdef int last = 2 * q - 1
    cdef double u = ((b - a) / (c - a)) * last
    cdef int k = <int>floor(u)
    if k < 0:
        k = 0
    elif k >= last:
        k = last
    elif u - k > 0.5:
        k += 1
    return k & 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef object dwt_embed_band(double[:, ::1] coeffs, Py_ssize_t half_h,
                          Py_ssize_t half_w, const Py_ssize_t[::1] positions,
                          const unsigned char[::1] bits, int q,
                          double span_epsilon, int level):
    """
    Встраивает сообщение в выбранные тройки одного уровня DWT.

    Args:
        coeffs: изменяемая упакованная матрица double с поддиапазонами уровня.
        half_h: высота каждого поддиапазона уровня.
        half_w: ширина каждого поддиапазона уровня.
        positions: заранее проверенные плоские адреса троек типа Py_ssize_t в порядке обхода.
        bits: непустой непрерывный массив unsigned char с битами, повторяемыми циклически.
        q: положительный параметр сетки квантования из 2*q узлов.
        span_epsilon: положительный порог вырожденности размаха тройки.
        level: номер уровня для сообщения об ошибке.
    """
    cdef Py_ssize_t i, r, c, length = bits.shape[0]
    cdef double values[3]
    cdef double target
    cdef int indices[3]
    cdef int median


    for i in range(positions.shape[0]):
        r = positions[i] // half_w
        c = positions[i] % half_w
        values[0] = coeffs[r + half_h, c]
        values[1] = coeffs[r, c + half_w]
        values[2] = coeffs[r + half_h, c + half_w]
        dwt_sort3(values, indices)
        if values[2] - values[0] <= span_epsilon:
            raise ValueError(
                f"DWT: degenerate selected triple at level={level}, "
                f"row={r}, col={c}; span <= span_epsilon. "
                "The strict profile does not change extrema or select a new key."
            )
        target = dwt_quantize(values[0], values[1], values[2], bits[i % length], q)
        median = indices[1]
        if median == 0:
            coeffs[r + half_h, c] = target
        elif median == 1:
            coeffs[r, c + half_w] = target
        else:
            coeffs[r + half_h, c + half_w] = target
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void dwt_extract_band(const double[:, ::1] coeffs, Py_ssize_t half_h,
                          Py_ssize_t half_w, const Py_ssize_t[::1] positions,
                          Py_ssize_t num_bits, int q, double span_epsilon,
                          long long[:, ::1] votes,
                          long long[::1] erased) noexcept nogil:
    """
    Накапливает голоса и стирания для троек одного уровня DWT.

    Args:
        coeffs: упакованная матрица double с поддиапазонами уровня.
        half_h: высота каждого поддиапазона уровня.
        half_w: ширина каждого поддиапазона уровня.
        positions: заранее проверенные плоские адреса троек типа Py_ssize_t.
        num_bits: положительное число битов; адрес i относится к биту i % num_bits независимо от стираний.
        q: положительный параметр сетки квантования.
        span_epsilon: положительный порог размаха, определяющий стирание тройки.
        votes: изменяемый массив long long формы (num_bits, 2) со счётчиками голосов за 0 и 1.
        erased: изменяемый массив long long длины num_bits со счётчиками стираний.
    """
    cdef Py_ssize_t i, r, c, bit_index
    cdef double values[3]
    cdef int indices[3]
    cdef int bit
    for i in range(positions.shape[0]):
        r = positions[i] // half_w
        c = positions[i] % half_w
        bit_index = i % num_bits
        values[0] = coeffs[r + half_h, c]
        values[1] = coeffs[r, c + half_w]
        values[2] = coeffs[r + half_h, c + half_w]
        dwt_sort3(values, indices)
        if values[2] - values[0] <= span_epsilon:
            erased[bit_index] += 1
        else:
            bit = dwt_decode(values[0], values[1], values[2], q)
            votes[bit_index, bit] += 1


cdef void svd_jacobi(double *A, double *V, double *sv, bint want_v) noexcept nogil:
    """
    Вычисляет сингулярные числа матрицы 4x4 методом Якоби.

    Args:
        A: изменяемый построчный буфер double длины 16 с исходной матрицей; столбцы преобразуются на месте.
        V: выходной построчный буфер double длины 16 для правых сингулярных векторов; используется при
            want_v=True.
        sv: выходной буфер double длины 4 для сингулярных чисел без сортировки.
        want_v: включает инициализацию V единичной матрицей и накопление правых вращений.
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
    Находит два наибольших сингулярных числа и индекс максимума.

    Args:
        sv: входной буфер double длины 4 с неотрицательными сингулярными числами в произвольном порядке.
        idx1: указатель на выходной индекс int первого максимального элемента.
        s1: указатель на выходное наибольшее значение double.
        s2: указатель на выходное второе по величине значение double.
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
    Встраивает бит QIM в старшее сингулярное число.

    Args:
        sigma: исходное неотрицательное старшее сингулярное число.
        bit: встраиваемый бит 0 или 1.
        delta: заранее проверенный положительный шаг квантования.

    Returns:
        quantized: новое значение double из решётки заданного бита; при равенстве расстояний выбирается первый
            кандидат.
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
    Извлекает бит QIM из сингулярного числа.

    Args:
        sigma: наблюдаемое старшее сингулярное число.
        delta: положительный шаг квантования, использованный при встраивании.

    Returns:
        bit: целое 0 или 1, определяемое чётностью ceil(sigma / delta) для неотрицательного sigma.
    """
    cdef double r = sigma / delta
    cdef int k = <int>ceil(r)
    return k % 2

cdef void power_iteration(double *A, double *u, double *v, double *sigma) noexcept nogil:
    """
    Оценивает старшее сингулярное число и векторы степенным методом.

    Args:
        A: входной построчный буфер double длины 16 с матрицей 4x4; не изменяется.
        u: выходной буфер double длины 4 для левого сингулярного вектора.
        v: выходной буфер double длины 4 для правого сингулярного вектора.
        sigma: указатель на выходную оценку старшего сингулярного числа double.
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
    Оценивает старшее сингулярное число степенным методом.

    Args:
        A: входной построчный буфер double длины 16 с матрицей 4x4; не изменяется.

    Returns:
        sigma: неотрицательная оценка double; при вырожденной итерации возвращается 0.
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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void embed_block(double[:, ::1] img, int by, int bx, int bit,
                       double delta) noexcept nogil:
    """
    Встраивает бит QIM в сингулярное число пространственного блока 4x4.

    Args:
        img: изменяемая двумерная матрица double с пространственными отсчётами изображения.
        by: неотрицательный номер полного блока 4x4 по вертикали.
        bx: неотрицательный номер полного блока 4x4 по горизонтали.
        bit: встраиваемый бит 0 или 1.
        delta: положительный шаг QIM; границы блока и параметры проверяет вызывающая сторона.
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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int extract_block(double[:, ::1] img, int by, int bx,
                        double delta) noexcept nogil:
    """
    Извлекает бит QIM из пространственного блока 4x4.

    Args:
        img: двумерная матрица double с принятым изображением; не изменяется.
        by: неотрицательный номер полного блока 4x4 по вертикали.
        bx: неотрицательный номер полного блока 4x4 по горизонтали.
        delta: положительный шаг QIM, использованный при встраивании; границы проверяет вызывающая сторона.

    Returns:
        bit: целое 0 или 1, извлечённое из оценки старшего сингулярного числа блока.
    """
    cdef double A[NN_SVD]
    cdef int i, j
    cdef int r0 = by * N_SVD
    cdef int c0 = bx * N_SVD

    for i in range(N_SVD):
        for j in range(N_SVD):
            A[i * N_SVD + j] = img[r0 + i, c0 + j]

    return qim_extract(power_iteration_sigma(A), delta)


from collections import namedtuple as contourlet_qim_namedtuple
from hashlib import sha256 as contourlet_qim_sha256
from numbers import Integral as contourlet_qim_Integral, Real as contourlet_qim_Real

import numpy as contourlet_qim_np

CONTOURLET_QIM_PROFILE_TYPE = contourlet_qim_namedtuple(
    'ContourletQIMProfile', 'levels delta channel key repetitions'
)
CONTOURLET_QIM_IMPLEMENTATION = 'native-lowpass-qim-cython-v1'


cpdef object contourlet_qim_profile(object levels=2, object delta=32.0, object channel=1, object key=0, object repetitions=3):
    """
    Создаёт неизменяемый профиль низкочастотного QIM.

    Args:
        levels: Число уровней пирамиды от 1 до 8.
        delta: Положительный шаг в единицах коэффициентов.
        channel: Индекс RGB-канала, по умолчанию 1 (зелёный).
        key: Общий ключ от 0 до 2**64-1; не подменяется seed пайплайна.
        repetitions: Число носителей на один бит, от 1 до 31.

    Returns:
        result: неизменяемый именованный кортеж с проверенными параметрами.
    """
    return CONTOURLET_QIM_PROFILE_TYPE(
        contourlet_qim_integer(levels, 'levels', 1, 8),
        contourlet_qim_positive_step(delta),
        contourlet_qim_integer(channel, 'channel', 0, 2),
        contourlet_qim_integer(key, 'key', 0, 2**64 - 1),
        contourlet_qim_integer(repetitions, 'repetitions', 1, 31),
    )

cpdef object contourlet_qim_checked_profile(object profile):
    """
    Проверяет профиль QIM или создаёт профиль по умолчанию.

    Args:
        profile: None для настроек по умолчанию либо профиль из contourlet_qim_profile.

    Returns:
        result: проверенный неизменяемый профиль.
    """
    if profile is None:
        return contourlet_qim_profile()
    if not isinstance(profile, CONTOURLET_QIM_PROFILE_TYPE):
        raise TypeError('profile must be created with contourlet_qim_profile')
    return contourlet_qim_profile(profile.levels, profile.delta, profile.channel,
                                 profile.key, profile.repetitions)

cpdef object contourlet_qim_options(object kwargs, object defaults, object payload_name):
    """
    Разбирает аргументы публичного интерфейса Contourlet.

    Args:
        kwargs: Аргументы embedding или extraction.
        defaults: Явный словарь параметров соответствующего метода.
        payload_name: watermark_bits либо num_bits.

    Returns:
        result: тройка: изображение, полезная нагрузка или её длина, профиль.
    """
    unknown = set(kwargs) - set(defaults)
    if unknown:
        raise TypeError('Unsupported Contourlet QIM options: ' + str(sorted(unknown)))
    args = dict(defaults)
    args.update(kwargs)
    if args['input_image'] is None or args[payload_name] is None:
        raise TypeError('input_image and ' + payload_name + ' are required')
    profile = contourlet_qim_profile(args['levels'], args['delta'], args['channel'],
                                    args['key'], args['repetitions'])
    return args['input_image'], args[payload_name], profile


CONTOURLET_QIM_ANALYSIS_TAPS = (
    0.037828455507, -0.023849465020, -0.110624404418,
    0.377402855613, 0.852698679009, 0.377402855613,
    -0.110624404418, -0.023849465020, 0.037828455507,
)


CONTOURLET_QIM_SYNTHESIS_TAPS = (
    -0.064538882629, -0.040689417609, 0.418092273222,
    0.788485616406, 0.418092273222, -0.040689417609,
    -0.064538882629,
)


cpdef object contourlet_qim_integer(object value, object name, object minimum=0, object maximum=None):
    """
    Проверяет целочисленный параметр QIM и его диапазон.

    Args:
        value: Проверяемое значение.
        name: Имя параметра для сообщения об ошибке.
        minimum: Включительная нижняя граница.
        maximum: Включительная верхняя граница либо None.

    Returns:
        result: целое число Python; bool и дробные значения не принимаются.
    """
    if isinstance(value, (bool, contourlet_qim_np.bool_)) or not isinstance(value, contourlet_qim_Integral):
        raise TypeError(f'{name} must be an integer, not {type(value).__name__}')
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f'{name} is outside [{minimum}, {maximum}]')
    return result


cpdef object contourlet_qim_positive_step(object value):
    """
    Проверяет положительный конечный шаг QIM.

    Args:
        value: Шаг в единицах низкочастотных коэффициентов.

    Returns:
        result: положительный конечный float, представимый вместе с четвертью шага.
    """
    if isinstance(value, (bool, contourlet_qim_np.bool_)) or not isinstance(value, contourlet_qim_Real):
        raise TypeError('delta must be a real number')
    result = float(value)
    if not contourlet_qim_np.isfinite(result) or result <= 0:
        raise ValueError('delta must be finite and positive')
    if result / 4 == 0 or result > contourlet_qim_np.finfo(contourlet_qim_np.float64).max / 4:
        raise ValueError('delta is outside the safe float64 representable range')
    return result


cpdef object contourlet_qim_real_array(object value, object name):
    """
    Преобразует конечные вещественные значения в float64.

    Args:
        value: Вещественный массив или совместимая последовательность.
        name: Имя массива для сообщения об ошибке.

    Returns:
        result: массив float64 без NaN/Infinity; создание копии не гарантируется.
    """
    array = contourlet_qim_np.asarray(value)
    if array.dtype.kind not in 'fiu':
        raise TypeError(f'{name} must contain real numbers')
    result = contourlet_qim_np.asarray(array, dtype=contourlet_qim_np.float64)
    if not contourlet_qim_np.all(contourlet_qim_np.isfinite(result)):
        raise ValueError(f'{name} contains NaN or infinity')
    return result


cpdef object contourlet_qim_binary_array(object value):
    """
    Проверяет одномерное бинарное сообщение.

    Args:
        value: Одномерный целочисленный или булев массив нулей и единиц.

    Returns:
        result: непрерывный одномерный uint8-массив.
    """
    array = contourlet_qim_np.asarray(value)
    if array.dtype.kind not in 'biu':
        raise TypeError('watermark_bits must be an integer or boolean array')
    if array.ndim != 1 or contourlet_qim_np.any((array != 0) & (array != 1)):
        raise ValueError('watermark_bits must be a one-dimensional binary array')
    return contourlet_qim_np.ascontiguousarray(array, dtype=contourlet_qim_np.uint8)


cpdef object contourlet_qim_checked_dither(object dither, object delta, object shape):
    """
    Проверяет смещения QIM и согласует их форму.

    Args:
        dither: Смещения для нулевого бита; для единицы используются противоположные.
        delta: Положительный шаг квантования.
        shape: Форма массива коэффициентов.

    Returns:
        result: массив float64 после broadcasting; модуль каждого элемента равен delta/4.
    """
    result = contourlet_qim_np.broadcast_to(contourlet_qim_real_array(dither, 'dither'), shape)
    if contourlet_qim_np.any(contourlet_qim_np.abs(result) != delta / 4):
        raise ValueError('This profile requires abs(dither) == delta / 4')
    return result


cpdef object contourlet_qim_nearest_quantize(object value, object delta):
    """
    Квантует значения к ближайшему кратному шагу.

    Args:
        value: Конечные вещественные значения.
        delta: Положительный шаг квантования.

    Returns:
        result: массив ближайших кратных delta; точные половины округляются к плюс бесконечности.
    """
    delta = contourlet_qim_positive_step(delta)
    array = contourlet_qim_real_array(value, 'coefficient')
    scaled = array / delta
    if not contourlet_qim_np.all(contourlet_qim_np.isfinite(scaled)) or contourlet_qim_np.any(contourlet_qim_np.abs(scaled) >= 2 ** 48):
        raise ValueError('Coefficient/step ratio is too large for reliable float64 QIM')
    return delta * contourlet_qim_np.floor(scaled + 0.5)


cpdef object contourlet_qim_embed_coefficients(object coefficients, object bits, object delta, object dither=None):
    """
    Встраивает биты QIM в отдельные коэффициенты.

    Args:
        coefficients: Одномерный конечный массив коэффициентов носителя.
        bits: По одному бинарному значению на каждый коэффициент.
        delta: Шаг каждого из двух квантователей.
        dither: Смещение нулевого бита плюс/минус delta/4; по умолчанию плюс delta/4.

    Returns:
        result: новый квантованный float64-массив; входной массив не изменяется.
    """
    delta = contourlet_qim_positive_step(delta)
    array = contourlet_qim_real_array(coefficients, 'coefficients')
    payload = contourlet_qim_binary_array(bits)
    if array.ndim != 1 or array.shape != payload.shape:
        raise ValueError('coefficients and bits must be one-dimensional and equally sized')
    d0 = contourlet_qim_checked_dither(delta / 4 if dither is None else dither, delta, array.shape)
    chosen = contourlet_qim_np.where(payload == 0, d0, -d0)
    return contourlet_qim_nearest_quantize(array + chosen, delta) - chosen


cpdef object contourlet_qim_extract_blocks(object coefficients, object delta, object repetitions=1, object dither=None):
    """
    Извлекает биты QIM по расстояниям до кодовых блоков.

    Args:
        coefficients: Одномерные коэффициенты, сгруппированные в последовательные L-блоки.
        delta: Общий шаг квантования.
        repetitions: Длина блока L, не правило голосования большинства.
        dither: Смещения нулевого бита для всех коэффициентов.

    Returns:
        result: по одному int8-биту на блок; равенство расстояний разрешается в пользу нуля.
    """
    delta = contourlet_qim_positive_step(delta)
    repetitions = contourlet_qim_integer(repetitions, 'repetitions', 1, 31)
    array = contourlet_qim_real_array(coefficients, 'coefficients')
    if array.ndim != 1 or array.size % repetitions:
        raise ValueError('The coefficient count must be divisible by repetitions')
    d0 = contourlet_qim_checked_dither(delta / 4 if dither is None else dither, delta, array.shape)
    zero = contourlet_qim_nearest_quantize(array + d0, delta) - d0
    one = contourlet_qim_nearest_quantize(array - d0, delta) + d0
    distance_zero = contourlet_qim_np.sum(((array - zero) / delta).reshape(-1, repetitions) ** 2, axis=1)
    distance_one = contourlet_qim_np.sum(((array - one) / delta).reshape(-1, repetitions) ** 2, axis=1)
    return contourlet_qim_np.ascontiguousarray(distance_one < distance_zero, dtype=contourlet_qim_np.int8)


cpdef object contourlet_qim_periodic_filter(object image, object taps):
    """
    Выполняет разделимую периодическую FIR-фильтрацию.

    Args:
        image: Непустой двумерный вещественный массив.
        taps: Одномерный фильтр нечётной длины с центром len(taps)//2.

    Returns:
        result: новый float64-массив исходной формы после периодической фильтрации по обеим осям.
    """
    result = contourlet_qim_real_array(image, 'image')
    if result.ndim != 2 or 0 in result.shape:
        raise ValueError('A nonempty two-dimensional array is required')
    weights = contourlet_qim_real_array(taps, 'taps')
    if weights.ndim != 1 or not len(weights) or len(weights) % 2 != 1:
        raise ValueError('An odd number of one-dimensional taps is required')
    for axis in (0, 1):
        filtered = contourlet_qim_np.zeros_like(result)
        for index, weight in enumerate(weights):
            filtered += weight * contourlet_qim_np.roll(result, index - len(weights) // 2, axis=axis)
        result = filtered
    return result


cpdef object contourlet_qim_lowpass_analysis(object image):
    """
    Выделяет низкочастотную область фильтром 9/7 с прореживанием.

    Args:
        image: Непустой двумерный массив с чётными размерами.

    Returns:
        result: непрерывный float64-массив с размерами, уменьшенными вдвое.
    """
    array = contourlet_qim_real_array(image, 'image')
    if array.ndim != 2 or 0 in array.shape or (array.shape[0] % 2 or array.shape[1] % 2):
        raise ValueError('Lowpass analysis requires positive even image dimensions')
    return contourlet_qim_np.ascontiguousarray(contourlet_qim_periodic_filter(array, CONTOURLET_QIM_ANALYSIS_TAPS)[::2, ::2])


cpdef object contourlet_qim_lowpass_synthesis(object coarse):
    """
    Восстанавливает низкочастотную область с удвоением размеров.

    Args:
        coarse: Непустая двумерная низкочастотная область.

    Returns:
        result: новый float64-массив с удвоенными размерами.
    """
    array = contourlet_qim_real_array(coarse, 'coarse')
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError('Synthesis requires a nonempty two-dimensional array')
    expanded = contourlet_qim_np.zeros((2 * array.shape[0], 2 * array.shape[1]), dtype=contourlet_qim_np.float64)
    expanded[::2, ::2] = array
    return contourlet_qim_periodic_filter(expanded, CONTOURLET_QIM_SYNTHESIS_TAPS)


cpdef object contourlet_qim_analyse(object image, object levels):
    """
    Вычисляет самую грубую низкочастотную область пирамиды.

    Args:
        image: Двумерный массив с размерами, кратными 2**levels.
        levels: Число уровней от 1 до 8.

    Returns:
        result: самая грубая низкочастотная область в float64.
    """
    levels = contourlet_qim_integer(levels, 'levels', 1, 8)
    array = contourlet_qim_real_array(image, 'image')
    if array.ndim != 2 or 0 in array.shape or (array.shape[0] % (2 ** levels) or array.shape[1] % (2 ** levels)):
        raise ValueError('Image dimensions must be positive and divisible by 2**levels')
    result = array
    for level in range(levels):
        result = contourlet_qim_lowpass_analysis(result)
    return result


cpdef object contourlet_qim_synthesise_delta(object coarse_delta, object levels):
    """
    Восстанавливает пространственную поправку низкочастотной ветви.

    Args:
        coarse_delta: Разность изменённой и исходной самой грубой области.
        levels: Число ступеней синтеза от 1 до 8.

    Returns:
        result: изменение float64 в полном пространственном разрешении.
    """
    levels = contourlet_qim_integer(levels, 'levels', 1, 8)
    result = contourlet_qim_real_array(coarse_delta, 'coarse_delta')
    for level in range(levels):
        result = contourlet_qim_lowpass_synthesis(result)
    return result


cpdef object contourlet_qim_carrier_layout(object total, object count, object key, object delta):
    """
    Выбирает носители и знаки дизеринга по ключу SHA-256.

    Args:
        total: Общее число доступных коэффициентов в построчном порядке.
        count: Число требуемых различных носителей.
        key: Общий беззнаковый 64-битный ключ; криптографическая стойкость не заявляется.
        delta: Шаг квантования.

    Returns:
        result: пара: плоские индексы int64 и смещения нулевого бита float64.
    """
    total = contourlet_qim_integer(total, 'total', 1)
    count = contourlet_qim_integer(count, 'count', 0, total)
    key = contourlet_qim_integer(key, 'key', 0, 2 ** 64 - 1)
    delta = contourlet_qim_positive_step(delta)
    if not count:
        return (contourlet_qim_np.empty(0, contourlet_qim_np.int64), contourlet_qim_np.empty(0, contourlet_qim_np.float64))
    prefix = b'DWARF-CT-QIM-v1\x00' + key.to_bytes(8, 'big')
    ranking = []
    for index in range(total):
        digest = contourlet_qim_sha256(prefix + b'P' + index.to_bytes(8, 'big')).digest()
        ranking.append((digest, index))
    ranking.sort()
    indices = contourlet_qim_np.empty(count, dtype=contourlet_qim_np.int64)
    signs = contourlet_qim_np.empty(count, dtype=contourlet_qim_np.float64)
    for rank in range(count):
        index = ranking[rank][1]
        indices[rank] = index
        digest = contourlet_qim_sha256(prefix + b'D' + index.to_bytes(8, 'big')).digest()
        signs[rank] = 1.0 if digest[0] & 1 else -1.0
    return (indices, signs * (delta / 4))


cpdef object contourlet_qim_embed_plane_float(object image, object watermark_bits, object profile=None):
    """
    Встраивает ЦВЗ в низкочастотную ветвь вещественной плоскости.

    Args:
        image: Двумерная плоскость с размерами, кратными 2**levels.
        watermark_bits: Одномерное бинарное сообщение с учётом repetitions носителей на бит.
        profile: Профиль из contourlet_qim_profile либо None для настроек по умолчанию.

    Returns:
        result: новая float64-плоскость до клиппинга и округления.
    """
    profile = contourlet_qim_checked_profile(profile)
    array = contourlet_qim_real_array(image, 'image')
    payload = contourlet_qim_binary_array(watermark_bits)
    coarse = contourlet_qim_analyse(array, profile.levels)
    indices, dither = contourlet_qim_carrier_layout(coarse.size, len(payload) * profile.repetitions, profile.key, profile.delta)
    target = contourlet_qim_embed_coefficients(coarse.ravel()[indices], contourlet_qim_np.repeat(payload, profile.repetitions), profile.delta, dither)
    change = contourlet_qim_np.zeros_like(coarse)
    change.ravel()[indices] = target - coarse.ravel()[indices]
    return array + contourlet_qim_synthesise_delta(change, profile.levels)


cpdef object contourlet_qim_extract_plane_float(object image, object num_bits, object profile=None):
    """
    Извлекает ЦВЗ из низкочастотной ветви вещественной плоскости.

    Args:
        image: Принятая плоскость с размерами, кратными 2**levels.
        num_bits: Известная длина извлекаемого сообщения.
        profile: Тот же профиль, что использовался при встраивании.

    Returns:
        result: непрерывный одномерный int8-массив оценённых битов.
    """
    profile = contourlet_qim_checked_profile(profile)
    coarse = contourlet_qim_analyse(image, profile.levels)
    num_bits = contourlet_qim_integer(num_bits, 'num_bits', 0)
    indices, dither = contourlet_qim_carrier_layout(coarse.size, num_bits * profile.repetitions, profile.key, profile.delta)
    return contourlet_qim_extract_blocks(coarse.ravel()[indices], profile.delta, profile.repetitions, dither)


cpdef object contourlet_qim_image_region(object image, object profile):
    """
    Проверяет изображение и выбирает рабочую область канала.

    Args:
        image: uint8-массив формы (H, W) или RGB (H, W, 3).
        profile: Профиль с выбранными levels и channel.

    Returns:
        result: четвёрка: проверенное изображение, плоскость канала, высота и ширина рабочей области.
    """
    profile = contourlet_qim_checked_profile(profile)
    if not isinstance(image, contourlet_qim_np.ndarray) or image.dtype != contourlet_qim_np.uint8:
        raise TypeError('input_image must be a numpy uint8 array; RGB is not BGR')
    if image.ndim == 2:
        plane = image
    elif image.ndim == 3 and image.shape[2] == 3:
        plane = image[:, :, profile.channel]
    else:
        raise ValueError('input_image must have shape HxW or HxWx3')
    scale = 2 ** profile.levels
    height = plane.shape[0] // scale * scale
    width = plane.shape[1] // scale * scale
    if min(height, width) < scale:
        raise ValueError('The image is too small for the selected number of levels')
    return (image, plane, height, width)


cpdef object contourlet_qim_capacity(object image, object profile=None):
    """
    Вычисляет ёмкость QIM с учётом повторений.

    Args:
        image: Корректное uint8-изображение RGB или в оттенках серого.
        profile: Профиль из contourlet_qim_profile либо None.

    Returns:
        result: максимальная длина сообщения с учётом repetitions носителей на бит.
    """
    profile = contourlet_qim_checked_profile(profile)
    array, plane, height, width = contourlet_qim_image_region(image, profile)
    return height // 2 ** profile.levels * (width // 2 ** profile.levels) // profile.repetitions


cpdef object contourlet_qim_round_uint8(object value):
    """
    Ограничивает пиксели диапазоном 0..255 и округляет в uint8.

    Args:
        value: Конечный вещественный массив пикселей.

    Returns:
        result: uint8-массив с ограничением диапазона 0..255 и явно заданным округлением.
    """
    array = contourlet_qim_real_array(value, 'pixels')
    return contourlet_qim_np.floor(contourlet_qim_np.clip(array, 0, 255) + 0.5).astype(contourlet_qim_np.uint8)


cpdef object contourlet_qim_embed_image(object image, object watermark_bits, object profile=None):
    """
    Встраивает ЦВЗ в изображение с итоговым преобразованием в uint8.

    Args:
        image: uint8-массив формы (H, W) или RGB (H, W, 3).
        watermark_bits: Одномерное бинарное сообщение, не превышающее ёмкость.
        profile: Явный профиль; при RGB меняется только выбранный канал.

    Returns:
        result: новый uint8-массив исходной формы. Клиппинг способен повредить сообщение. Скрытых повторных
            проходов или подбора шага по содержимому нет.
    """
    profile = contourlet_qim_checked_profile(profile)
    array, plane, height, width = contourlet_qim_image_region(image, profile)
    marked = contourlet_qim_embed_plane_float(plane[:height, :width], watermark_bits, profile)
    result = array.copy()
    if result.ndim == 2:
        result[:height, :width] = contourlet_qim_round_uint8(marked)
    else:
        result[:height, :width, profile.channel] = contourlet_qim_round_uint8(marked)
    return result


cpdef object contourlet_qim_extract_image(object image, object num_bits, object profile=None):
    """
    Выполняет слепое извлечение ЦВЗ из uint8-изображения.

    Args:
        image: Принятое uint8-изображение без исходного контейнера.
        num_bits: Точная требуемая длина сообщения; не определяется по исходным битам.
        profile: Те же levels, delta, channel, key и repetitions, что при встраивании.

    Returns:
        result: непрерывный одномерный int8-массив со значениями 0 и 1.
    """
    profile = contourlet_qim_checked_profile(profile)
    array, plane, height, width = contourlet_qim_image_region(image, profile)
    return contourlet_qim_extract_plane_float(plane[:height, :width], num_bits, profile)


cpdef cnp.ndarray dft_validate_rgb_uint8(object image):
    """
    Проверяет тип и форму RGB-изображения для DFT.

    Args:
        image: входное RGB-изображение с dtype uint8 и формой (H, W, 3).

    Returns:
        result: C-contiguous RGB-массив uint8 формы (H, W, 3).
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"input_image must have shape (H, W, 3), got {arr.shape}")
    if arr.dtype != np.uint8:
        raise TypeError(f"input_image must have dtype uint8, got {arr.dtype}")
    return np.ascontiguousarray(arr)


cpdef cnp.ndarray dft_validate_bits(object bits):
    """
    Проверяет бинарное сообщение для DFT.

    Args:
        bits: одномерный массив uint8 со значениями 0 и 1.

    Returns:
        result: C-contiguous массив uint8 с проверенными битами ЦВЗ.
    """
    arr = np.asarray(bits)
    if arr.ndim != 1:
        raise ValueError(f"watermark_bits must be one-dimensional, got {arr.shape}")
    if arr.dtype != np.uint8:
        raise TypeError(f"watermark_bits must have dtype uint8, got {arr.dtype}")
    if arr.size == 0:
        raise ValueError("watermark_bits must not be empty")
    if np.any((arr != 0) & (arr != 1)):
        raise ValueError("watermark_bits must contain only 0 and 1")
    return np.ascontiguousarray(arr)


cpdef cnp.ndarray dft_rgb_to_y(object rgb):
    """
    Вычисляет яркостную компоненту RGB-изображения.

    Args:
        rgb: RGB-изображение формы (H, W, 3).

    Returns:
        y: яркостная компонента float64 формы (H, W).
    """
    arr = np.asarray(rgb)
    return np.ascontiguousarray(
        0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2],
        dtype=np.float64,
    )


cpdef cnp.ndarray dft_apply_luma_delta(object rgb, object old_y, object new_y):
    """
    Переносит изменение яркости DFT-встраивания в RGB.

    Args:
        rgb: исходное RGB-изображение uint8 формы (H, W, 3).
        old_y: исходная яркостная компонента формы (H, W).
        new_y: изменённая яркостная компонента формы (H, W).

    Returns:
        result: RGB-изображение uint8 формы (H, W, 3) после переноса яркостной поправки.
    """
    delta = np.asarray(new_y, dtype=np.float64) - np.asarray(old_y, dtype=np.float64)
    out = np.asarray(rgb, dtype=np.float64) + delta[..., None]
    return np.ascontiguousarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


cpdef int dft_binary_parity(int value):
    """
    Вычисляет чётность числа единичных битов целого числа.

    Args:
        value: неотрицательное целое число.

    Returns:
        parity: 0 при чётном числе единичных битов и 1 при нечётном.
    """
    cdef int parity = 0
    while value:
        parity ^= value & 1
        value >>= 1
    return parity


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray dft_convolutional_encode(object bits):
    """
    Кодирует сообщение свёрточным кодом 1/3 с памятью 6.

    Args:
        bits: одномерный массив информационных битов.

    Returns:
        coded: массив uint8 длины 3 * (L + 6), где L является длиной сообщения.
    """
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    cdef int state = 0
    cdef int u
    cdef int reg
    cdef int index = 0
    cdef int i
    cdef int total = int(data.size) + 6
    coded = np.empty(3 * total, dtype=np.uint8)
    for i in range(total):
        if i < data.size:
            u = int(data[i])
        else:
            u = 0
        reg = (state << 1) | u
        coded[index] = dft_binary_parity(reg & 0o171)
        coded[index + 1] = dft_binary_parity(reg & 0o133)
        coded[index + 2] = dft_binary_parity(reg & 0o165)
        index += 3
        state = ((state << 1) | u) & 63
    return np.ascontiguousarray(coded)


cpdef cnp.ndarray dft_convolutional_decode_soft_batch(object values, int message_length):
    """
    Декодирует пакет мягких последовательностей алгоритмом Витерби.

    Args:
        values: двумерный массив мягких оценок формы (N, 3 * (L + 6)).
        message_length: длина исходного сообщения в битах.

    Returns:
        decoded: массив uint8 формы (N, L) с восстановленными сообщениями.
    """
    matrix_array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if matrix_array.ndim != 2:
        raise ValueError("values must have shape (N, coded_length)")
    cdef int batch = matrix_array.shape[0]
    cdef int steps = message_length + 6
    cdef int need = 3 * steps
    cdef int n_states = 64
    cdef int t
    cdef int row
    cdef int state
    cdef int u
    cdef int next_state
    cdef int reg
    cdef int b0
    cdef int b1
    cdef int b2
    cdef int max_u
    cdef int traceback_state
    cdef double metric
    cdef double negative = -1.0e300
    if message_length <= 0:
        raise ValueError("message_length must be > 0")
    if matrix_array.shape[1] < need:
        raise ValueError("Not enough soft values for conv_r13_m6 decoding")
    metrics_array = np.empty((batch, n_states), dtype=np.float64)
    next_metrics_array = np.empty((batch, n_states), dtype=np.float64)
    previous_state_array = np.full((steps, batch, n_states), -1, dtype=np.int16)
    previous_bit_array = np.full((steps, batch, n_states), -1, dtype=np.int8)
    decoded_array = np.empty((batch, message_length), dtype=np.uint8)
    cdef double[:, ::1] matrix = matrix_array
    cdef double[:, ::1] metrics = metrics_array
    cdef double[:, ::1] next_metrics = next_metrics_array
    cdef short[:, :, ::1] previous_state = previous_state_array
    cdef signed char[:, :, ::1] previous_bit = previous_bit_array
    cdef unsigned char[:, ::1] decoded = decoded_array
    cdef double[:, ::1] temporary_metrics
    for row in range(batch):
        for state in range(n_states):
            metrics[row, state] = negative
        metrics[row, 0] = 0.0
    for t in range(steps):
        for row in range(batch):
            for state in range(n_states):
                next_metrics[row, state] = negative
        max_u = 2 if t < message_length else 1
        for state in range(n_states):
            for u in range(max_u):
                reg = (state << 1) | u
                b0 = dft_binary_parity(reg & 0o171)
                b1 = dft_binary_parity(reg & 0o133)
                b2 = dft_binary_parity(reg & 0o165)
                next_state = ((state << 1) | u) & 63
                for row in range(batch):
                    if metrics[row, state] <= negative / 2.0:
                        continue
                    metric = metrics[row, state]
                    metric += matrix[row, 3 * t] * (1.0 if b0 else -1.0)
                    metric += matrix[row, 3 * t + 1] * (1.0 if b1 else -1.0)
                    metric += matrix[row, 3 * t + 2] * (1.0 if b2 else -1.0)
                    if metric > next_metrics[row, next_state]:
                        next_metrics[row, next_state] = metric
                        previous_state[t, row, next_state] = state
                        previous_bit[t, row, next_state] = u
        temporary_metrics = metrics
        metrics = next_metrics
        next_metrics = temporary_metrics
    for row in range(batch):
        traceback_state = 0
        for t in range(steps - 1, -1, -1):
            if previous_state[t, row, traceback_state] < 0:
                raise ValueError("Viterbi traceback failed for conv_r13_m6")
            if t < message_length:
                decoded[row, t] = previous_bit[t, row, traceback_state]
            traceback_state = previous_state[t, row, traceback_state]
    return np.ascontiguousarray(decoded_array)


cpdef cnp.ndarray dft_ecc_encode(object bits, str mode):
    """
    Кодирует сообщение выбранным кодом коррекции ошибок.

    Args:
        bits: одномерный массив битов исходного сообщения.
        mode: режим conv_r13_m6, repeat3 или none.

    Returns:
        coded: одномерный массив uint8 с закодированной последовательностью.
    """
    data = np.asarray(bits, dtype=np.uint8)
    if mode == "none":
        return np.ascontiguousarray(data.copy())
    if mode == "repeat3":
        return np.ascontiguousarray(np.repeat(data, 3), dtype=np.uint8)
    if mode == "conv_r13_m6":
        return dft_convolutional_encode(data)
    raise ValueError("ecc_mode must be 'conv_r13_m6', 'repeat3' or 'none'")


cpdef cnp.ndarray dft_ecc_decode_soft_batch(object values, int message_length, str mode):
    """
    Декодирует пакет мягких оценок выбранным кодом коррекции ошибок.

    Args:
        values: двумерный массив мягких значений формы (N, coded_length).
        message_length: длина исходного сообщения в битах.
        mode: режим conv_r13_m6, repeat3 или none.

    Returns:
        decoded: массив uint8 формы (N, message_length).
    """
    matrix = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if matrix.ndim != 2:
        raise ValueError("values must have shape (N, coded_length)")
    if mode == "none":
        if matrix.shape[1] < message_length:
            raise ValueError("Not enough soft values for ecc_mode='none'")
        return np.ascontiguousarray((matrix[:, :message_length] >= 0.0).astype(np.uint8))
    if mode == "repeat3":
        need = 3 * message_length
        if matrix.shape[1] < need:
            raise ValueError("Not enough soft values for repeat3 decoding")
        grouped = matrix[:, :need].reshape(matrix.shape[0], message_length, 3)
        return np.ascontiguousarray((grouped.sum(axis=2) >= 0.0).astype(np.uint8))
    if mode == "conv_r13_m6":
        return dft_convolutional_decode_soft_batch(matrix, message_length)
    raise ValueError("ecc_mode must be 'conv_r13_m6', 'repeat3' or 'none'")


cpdef cnp.ndarray dft_coded_antipodal_from_messages(object messages, str ecc_mode):
    """
    Преобразует пакет сообщений в антиподальные кодовые слова ECC.

    Args:
        messages: двумерный массив битов формы (N, L).
        ecc_mode: режим ECC, совпадающий с режимом встраивания.

    Returns:
        antipodal: массив float64 формы (N, coded_length) со значениями -1 и +1.
    """
    data_array = np.ascontiguousarray(np.asarray(messages, dtype=np.uint8))
    if data_array.ndim != 2:
        raise ValueError("messages must have shape (N, L)")
    if ecc_mode == "none":
        return np.ascontiguousarray(np.where(data_array > 0, 1.0, -1.0), dtype=np.float64)
    if ecc_mode == "repeat3":
        coded = np.repeat(data_array, 3, axis=1)
        return np.ascontiguousarray(np.where(coded > 0, 1.0, -1.0), dtype=np.float64)
    if ecc_mode != "conv_r13_m6":
        raise ValueError("ecc_mode must be 'conv_r13_m6', 'repeat3' or 'none'")
    cdef int batch = data_array.shape[0]
    cdef int message_length = data_array.shape[1]
    cdef int steps = message_length + 6
    cdef int coded_length = 3 * steps
    cdef int row
    cdef int t
    cdef int state
    cdef int u
    cdef int reg
    coded_array = np.empty((batch, coded_length), dtype=np.uint8)
    cdef unsigned char[:, ::1] data = data_array
    cdef unsigned char[:, ::1] coded_view = coded_array
    for row in range(batch):
        state = 0
        for t in range(steps):
            u = data[row, t] if t < message_length else 0
            reg = (state << 1) | u
            coded_view[row, 3 * t] = dft_binary_parity(reg & 0o171)
            coded_view[row, 3 * t + 1] = dft_binary_parity(reg & 0o133)
            coded_view[row, 3 * t + 2] = dft_binary_parity(reg & 0o165)
            state = ((state << 1) | u) & 63
    return np.ascontiguousarray(np.where(coded_array > 0, 1.0, -1.0), dtype=np.float64)


cpdef cnp.ndarray dft_interleaver_indices(int length, int seed):
    """
    Строит детерминированную перестановку позиций кодового слова.

    Args:
        length: длина кодового слова.
        seed: целочисленное зерно интерливера.

    Returns:
        indices: массив int64, задающий порядок передачи кодовых битов.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    indices = np.arange(length, dtype=np.int64)
    cdef unsigned int state = <unsigned int> seed
    cdef int i
    cdef int j
    cdef cnp.int64_t temp
    cdef cnp.int64_t[::1] view = indices
    if state == 0:
        state = 2463534242
    for i in range(length - 1, 0, -1):
        state ^= state << 13
        state ^= state >> 17
        state ^= state << 5
        j = <int>(state % <unsigned int>(i + 1))
        temp = view[i]
        view[i] = view[j]
        view[j] = temp
    return np.ascontiguousarray(indices, dtype=np.int64)


cpdef cnp.ndarray dft_interleave_coded_bits(object bits, int seed):
    """
    Переставляет кодовые биты перед Manchester-кодированием.

    Args:
        bits: одномерный массив кодовых битов.
        seed: зерно детерминированного интерливера.

    Returns:
        interleaved: кодовые биты в порядке передачи по угловым частотам.
    """
    data = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8).reshape(-1))
    indices = dft_interleaver_indices(int(data.size), seed)
    return np.ascontiguousarray(data[indices], dtype=np.uint8)


cpdef cnp.ndarray dft_deinterleave_soft_matrix(object values, int seed):
    """
    Восстанавливает исходный порядок мягких оценок ECC.

    Args:
        values: двумерный массив мягких оценок формы (N, coded_length).
        seed: зерно интерливера, совпадающее с этапом встраивания.

    Returns:
        restored: массив float64 формы (N, coded_length) в порядке ECC-декодера.
    """
    matrix = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if matrix.ndim != 2:
        raise ValueError("values must have shape (N, coded_length)")
    indices = dft_interleaver_indices(int(matrix.shape[1]), seed)
    restored = np.empty_like(matrix)
    restored[:, indices] = matrix
    return np.ascontiguousarray(restored, dtype=np.float64)


cpdef cnp.ndarray dft_manchester_encode(object bits):
    """
    Выполняет Manchester-кодирование бинарного сообщения.

    Args:
        bits: одномерный массив кодовых битов, где 1 кодируется парой (1, 0), а 0 парой (0, 1).

    Returns:
        encoded: массив float64 удвоенной длины с Manchester-последовательностью.
    """
    b = np.asarray(bits, dtype=np.uint8)
    out = np.empty(2 * b.size, dtype=np.float64)
    out[0::2] = b
    out[1::2] = 1 - b
    return np.ascontiguousarray(out)


_DFT_LPM_GEOMETRY_CACHE = {}
_DFT_LPM_GEOMETRY_CACHE_LIMIT = 8


cpdef object dft_forward_lpm_geometry(int height, int width, int rho_samples, int theta_samples,
                                      double omega_min, double omega_max):
    """
    Строит и кэширует геометрию прямого log-polar отображения.

    Args:
        height: высота центрированного спектра.
        width: ширина центрированного спектра.
        rho_samples: число отсчётов по логарифмическому радиусу.
        theta_samples: число отсчётов на угловой полуокружности.
        omega_min: минимальная нормированная радиальная частота.
        omega_max: максимальная нормированная радиальная частота.

    Returns:
        geometry: словарь с индексами исходных отсчётов, log-polar bins и числами попаданий.
    """
    key = (
        int(height),
        int(width),
        int(rho_samples),
        int(theta_samples),
        float(omega_min),
        float(omega_max),
    )
    cached = _DFT_LPM_GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached
    cy = height // 2
    cx = width // 2
    yy = (np.arange(height, dtype=np.float64) - cy) / max(cy, 1)
    xx = (np.arange(width, dtype=np.float64) - cx) / max(cx, 1)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    omega = np.sqrt(X * X + Y * Y) / np.sqrt(2.0)
    theta = np.mod(np.arctan2(Y, X), np.pi)
    valid = (omega >= omega_min) & (omega <= omega_max)
    if not np.any(valid):
        raise ValueError("No Fourier samples fall inside the requested log-polar range")
    source_indices = np.flatnonzero(valid.reshape(-1)).astype(np.int64)
    omega_valid = omega.reshape(-1)[source_indices]
    theta_valid = theta.reshape(-1)[source_indices]
    lr = np.log(np.maximum(omega_valid, 1e-300)) - np.log(omega_min)
    lr /= np.log(omega_max) - np.log(omega_min)
    r_idx = np.rint(lr * (rho_samples - 1)).astype(np.int64)
    t_idx = np.rint(theta_valid / np.pi * theta_samples).astype(np.int64)
    t_idx %= theta_samples
    flat_idx = np.ascontiguousarray(r_idx * theta_samples + t_idx, dtype=np.int64)
    counts_flat = np.bincount(flat_idx, minlength=rho_samples * theta_samples).astype(np.int64)
    counts = counts_flat.reshape(rho_samples, theta_samples)
    occupied_rows = [np.flatnonzero(counts[r] > 0).astype(np.int64) for r in range(rho_samples)]
    geometry = {
        "source_indices": np.ascontiguousarray(source_indices, dtype=np.int64),
        "flat_idx": flat_idx,
        "counts_flat": np.ascontiguousarray(counts_flat, dtype=np.int64),
        "occupied_rows": occupied_rows,
    }
    if len(_DFT_LPM_GEOMETRY_CACHE) >= _DFT_LPM_GEOMETRY_CACHE_LIMIT:
        oldest_key = next(iter(_DFT_LPM_GEOMETRY_CACHE))
        _DFT_LPM_GEOMETRY_CACHE.pop(oldest_key, None)
    _DFT_LPM_GEOMETRY_CACHE[key] = geometry
    return geometry


cpdef cnp.ndarray dft_forward_lpm(object magnitude, int rho_samples, int theta_samples,
                                  double omega_min, double omega_max):
    """
    Выполняет прямое log-polar отображение амплитуды DFT.

    Args:
        magnitude: центрированная амплитуда двумерного DFT формы (H, W).
        rho_samples: число отсчётов по логарифмическому радиусу.
        theta_samples: число отсчётов на независимой угловой полуокружности [0, pi).
        omega_min: минимальная нормированная радиальная частота.
        omega_max: максимальная нормированная радиальная частота.

    Returns:
        lpm: log-polar представление float64 формы (rho_samples, theta_samples).
    """
    source = np.asarray(magnitude, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("magnitude must be two-dimensional")
    H, W = source.shape
    geometry = dft_forward_lpm_geometry(
        H,
        W,
        rho_samples,
        theta_samples,
        omega_min,
        omega_max,
    )
    vals = source.reshape(-1)[geometry["source_indices"]]
    sums = np.bincount(
        geometry["flat_idx"],
        weights=vals,
        minlength=rho_samples * theta_samples,
    )
    counts_flat = geometry["counts_flat"]
    lpm_flat = np.zeros(rho_samples * theta_samples, dtype=np.float64)
    occupied_flat = counts_flat > 0
    lpm_flat[occupied_flat] = sums[occupied_flat] / counts_flat[occupied_flat]
    lpm = lpm_flat.reshape(rho_samples, theta_samples)
    x = np.arange(theta_samples, dtype=np.float64)
    for r in range(rho_samples):
        xp = geometry["occupied_rows"][r]
        n = int(xp.size)
        if n == 0:
            continue
        if n == 1:
            lpm[r, :] = lpm[r, xp[0]]
            continue
        fp = lpm[r, xp]
        xp_ext = np.concatenate((xp - theta_samples, xp, xp + theta_samples))
        fp_ext = np.concatenate((fp, fp, fp))
        lpm[r, :] = np.interp(x, xp_ext, fp_ext)
    return np.ascontiguousarray(lpm)


cpdef cnp.ndarray dft_bilinear_periodic(object arr, object rho_f, object theta_f):
    """
    Интерполирует log-polar массив с периодической угловой координатой.

    Args:
        arr: двумерный log-polar массив формы (R, T).
        rho_f: вещественные координаты по радиальной оси.
        theta_f: вещественные координаты по угловой оси.

    Returns:
        values: интерполированные значения для заданных координат.
    """
    source = np.asarray(arr, dtype=np.float64)
    rf_input = np.asarray(rho_f, dtype=np.float64)
    tf_input = np.asarray(theta_f, dtype=np.float64)
    R, T = source.shape
    rf = np.clip(rf_input, 0.0, R - 1.0)
    tf = np.mod(tf_input, T)
    r0 = np.floor(rf).astype(np.int64)
    r1 = np.minimum(r0 + 1, R - 1)
    t0 = np.floor(tf).astype(np.int64) % T
    t1 = (t0 + 1) % T
    ar = rf - r0
    at = tf - np.floor(tf)
    a00 = source[r0, t0]
    a01 = source[r0, t1]
    a10 = source[r1, t0]
    a11 = source[r1, t1]
    values = (
        (1.0 - ar) * (1.0 - at) * a00
        + (1.0 - ar) * at * a01
        + ar * (1.0 - at) * a10
        + ar * at * a11
    )
    return np.ascontiguousarray(values)


cpdef cnp.ndarray dft_inverse_lpm_backward(object lpm, object out_shape,
                                           double omega_min, double omega_max):
    """
    Выполняет обратное log-polar отображение в декартову сетку.

    Args:
        lpm: двумерный log-polar массив формы (R, T) с угловой областью [0, pi).
        out_shape: требуемая форма выходного массива (H, W).
        omega_min: положительная минимальная частота log-polar сетки.
        omega_max: максимальная частота сетки, строго больше omega_min.

    Returns:
        out: C-contiguous массив float64 формы (H, W) с интерполированными значениями внутри частотного кольца и
            нулями вне него.
    """
    source = np.asarray(lpm, dtype=np.float64)
    H, W = out_shape
    cy = H // 2
    cx = W // 2
    yy = (np.arange(H, dtype=np.float64) - cy) / max(cy, 1)
    xx = (np.arange(W, dtype=np.float64) - cx) / max(cx, 1)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    omega = np.sqrt(X * X + Y * Y) / np.sqrt(2.0)
    theta = np.mod(np.arctan2(Y, X), np.pi)
    R, T = source.shape
    valid = (omega >= omega_min) & (omega <= omega_max)
    out = np.zeros((H, W), dtype=np.float64)
    if not np.any(valid):
        return np.ascontiguousarray(out)
    rho_f = (
        (np.log(np.maximum(omega[valid], 1e-300)) - np.log(omega_min))
        / (np.log(omega_max) - np.log(omega_min))
        * (R - 1)
    )
    theta_f = theta[valid] / np.pi * T
    out[valid] = dft_bilinear_periodic(source, rho_f, theta_f)
    return np.ascontiguousarray(out)


cpdef object dft_centered_fft_with_optional_padding(object y, int pad_factor):
    """
    Вычисляет центрированный DFT с симметричным нулевым дополнением.

    Args:
        y: яркостная компонента float64 формы (H, W).
        pad_factor: целочисленный коэффициент увеличения каждой размерности перед DFT.

    Returns:
        spectrum: центрированный комплексный DFT дополненного изображения.
        original_shape: исходная форма яркостной компоненты (H, W).
    """
    if pad_factor < 1:
        raise ValueError("pad_factor must be >= 1")
    source = np.asarray(y, dtype=np.float64)
    H, W = source.shape
    PH = int(H * pad_factor)
    PW = int(W * pad_factor)
    if PH < H or PW < W:
        raise ValueError("pad_factor produced a padded shape smaller than the image")
    padded = np.zeros((PH, PW), dtype=np.float64)
    y0 = (PH - H) // 2
    x0 = (PW - W) // 2
    padded[y0:y0 + H, x0:x0 + W] = source
    spectrum = np.fft.fftshift(np.fft.fft2(padded))
    return np.ascontiguousarray(spectrum, dtype=np.complex128), (H, W)


cpdef cnp.ndarray dft_centered_magnitude_with_optional_padding(object y, int pad_factor):
    """
    Вычисляет модуль центрированного DFT с нулевым дополнением.

    Args:
        y: яркостная компонента float64 формы (H, W).
        pad_factor: целочисленный коэффициент увеличения каждой размерности перед DFT.

    Returns:
        magnitude: модуль центрированного DFT дополненного изображения.
    """
    if pad_factor < 1:
        raise ValueError("pad_factor must be >= 1")
    source = np.asarray(y, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("y must be two-dimensional")
    H, W = source.shape
    PH = int(H * pad_factor)
    PW = int(W * pad_factor)
    if PH < H or PW < W:
        raise ValueError("pad_factor produced a padded shape smaller than the image")
    spectrum = np.fft.fftshift(np.fft.fft2(source, s=(PH, PW)))
    return np.ascontiguousarray(np.abs(spectrum), dtype=np.float64)


cpdef cnp.ndarray dft_crop_center(object arr, object shape):
    """
    Вырезает центральную область двумерного массива.

    Args:
        arr: входной двумерный массив.
        shape: требуемая форма результата (H, W).

    Returns:
        cropped: центральный C-contiguous фрагмент заданной формы.
    """
    source = np.asarray(arr)
    H, W = shape
    AH, AW = source.shape
    y0 = (AH - H) // 2
    x0 = (AW - W) // 2
    return np.ascontiguousarray(source[y0:y0 + H, x0:x0 + W])


cpdef object dft_rt_invariant_from_image(object y, int rho_samples, int theta_samples,
                                         double omega_min, double omega_max, int pad_factor):
    """
    Строит DFT/LPM-представление для встраивания ЦВЗ.

    Args:
        y: яркостная компонента изображения формы (H, W).
        rho_samples: число отсчётов по логарифмическому радиусу.
        theta_samples: число отсчётов на независимой угловой полуокружности.
        omega_min: минимальная частота полной log-polar сетки.
        omega_max: максимальная частота полной log-polar сетки.
        pad_factor: коэффициент нулевого дополнения перед двумерным DFT.

    Returns:
        data: словарь со спектром, фазами, LPM и RT-инвариантным представлением.
    """
    spectrum, original_shape = dft_centered_fft_with_optional_padding(y, pad_factor)
    magnitude = np.abs(spectrum)
    lpm = dft_forward_lpm(magnitude, rho_samples, theta_samples, omega_min, omega_max)
    angular_fft = np.fft.fft(lpm, axis=1)
    return {
        "F": spectrum,
        "original_shape": original_shape,
        "magnitude": magnitude,
        "phase_F": np.angle(spectrum),
        "lpm": lpm,
        "angular_fft": angular_fft,
        "rt_magnitude": np.abs(angular_fft),
        "rt_phase": np.angle(angular_fft),
    }


cpdef cnp.ndarray dft_rt_magnitude_from_image(object y, int rho_samples, int theta_samples,
                                                double omega_min, double omega_max, int pad_factor):
    """
    Вычисляет модуль DFT/LPM-представления для извлечения ЦВЗ.

    Args:
        y: яркостная компонента изображения формы (H, W).
        rho_samples: число отсчётов по логарифмическому радиусу.
        theta_samples: число отсчётов на независимой угловой полуокружности.
        omega_min: минимальная частота полной log-polar сетки.
        omega_max: максимальная частота полной log-polar сетки.
        pad_factor: коэффициент нулевого дополнения перед двумерным DFT.

    Returns:
        rt_magnitude: массив float64 формы (rho_samples, theta_samples) без вычисления ненужных фаз.
    """
    magnitude = dft_centered_magnitude_with_optional_padding(y, pad_factor)
    lpm = dft_forward_lpm(magnitude, rho_samples, theta_samples, omega_min, omega_max)
    return np.ascontiguousarray(np.abs(np.fft.fft(lpm, axis=1)), dtype=np.float64)


cpdef int dft_rho_index_for_omega(double omega, int rho_samples,
                                  double omega_min, double omega_max):
    """
    Переводит радиальную частоту в индекс логарифмической оси rho.

    Args:
        omega: нормированная радиальная частота.
        rho_samples: число отсчётов по логарифмической радиальной оси.
        omega_min: минимальная частота оси.
        omega_max: максимальная частота оси.

    Returns:
        index: ближайший допустимый индекс rho.
    """
    f = (np.log(omega) - np.log(omega_min)) / (np.log(omega_max) - np.log(omega_min))
    return int(np.rint(np.clip(f, 0.0, 1.0) * (rho_samples - 1)))


cpdef cnp.ndarray dft_scale_delta_to_psnr(object base_y, object delta, object target_psnr):
    """
    Масштабирует пространственную добавку до заданного PSNR.

    Args:
        base_y: исходная яркостная плоскость; при заданном target_psnr лишь приводится к float64, её значения в
            расчёте не используются.
        delta: двумерная пространственная добавка ЦВЗ, приводимая к float64.
        target_psnr: положительный конечный PSNR в децибелах для диапазона 0..255 либо None без масштабирования.

    Returns:
        scaled: C-contiguous массив float64 с заданной средней мощностью; при мощности не больше 1e-24 добавка
            сохраняется.
    """
    source = np.asarray(delta, dtype=np.float64)
    if target_psnr is None:
        return np.ascontiguousarray(source)
    target = float(target_psnr)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("target_psnr must be a positive finite number or None")
    np.asarray(base_y, dtype=np.float64)
    mse_target = (255.0 ** 2) / (10.0 ** (target / 10.0))
    mse_delta = float(np.mean(source * source))
    if mse_delta <= 1e-24:
        return np.ascontiguousarray(source)
    return np.ascontiguousarray(source * np.sqrt(mse_target / mse_delta))
