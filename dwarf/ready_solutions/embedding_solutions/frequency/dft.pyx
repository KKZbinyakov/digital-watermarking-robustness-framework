"""
Mauro Barni, Franco Bartolini, Alessandro De Rosa, Alessandro Piva
Optimum Decoding and Detection of Multiplicative Watermarks (2003)
https://doi.org/10.1109/TSP.2003.809371
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log1p, expm1, log10, isfinite

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    dft_validate_image, dft_validate_bits, dft_validate_key, dft_validate_strength,
    dft_layout, dft_key_material, dft_build_mask,
    dft_fit_weibull, dft_render_pixels,
)

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint _embed_core(cnp.complex128_t[:, ::1] spectrum,
                      const cnp.intp_t[:, ::1] coords,
                      const cnp.intp_t[::1] zones,
                      const cnp.intp_t[::1] permutation,
                      const double[::1] carrier, const cnp.uint8_t[::1] bits,
                      double nominal_strength) noexcept nogil:
    """
    Встраивает биты ЦВЗ в модули DFT-коэффициентов и сохраняет сопряжённую симметрию.

    Args:
        spectrum: изменяемая матрица complex128 формы (512, 512), без fftshift.
        coords: знаковые координаты независимых коэффициентов intp формы (18960, 2).
        zones: номера 16 областей Вейбулла в порядке coords.
        permutation: ключевая перестановка индексов независимых коэффициентов.
        carrier: ключевая несущая float64 длины 18960 в порядке групп.
        bits: проверенный непрерывный массив uint8 из 64 битов 0 и 1.
        nominal_strength: положительная сила изменения коэффициентов до маски, не выше 0.95.

    Returns:
        bool: True при успехе; False, если область или битовая группа содержит только нулевые коэффициенты.
    """
    cdef bint nonzero_zones[16]
    cdef bint nonzero_groups[64]
    cdef int j, i, k, v, u, row
    cdef double gain
    cdef cnp.complex128_t value
    for i in range(16):
        nonzero_zones[i] = False
    for k in range(64):
        nonzero_groups[k] = False
    for j in range(18960):
        i = permutation[j]
        k = j // 296
        if k > 63:
            k = 63
        v, u = coords[i, 0], coords[i, 1]
        row = v + 512 if v < 0 else v
        value = spectrum[row, u]
        if value.real != 0.0 or value.imag != 0.0:
            nonzero_zones[zones[i]] = True
            nonzero_groups[k] = True
    for i in range(16):
        if not nonzero_zones[i]:
            return False
    for k in range(64):
        if not nonzero_groups[k]:
            return False
    for j in range(18960):
        i = permutation[j]
        k = j // 296
        if k > 63:
            k = 63
        v, u = coords[i, 0], coords[i, 1]
        row = v + 512 if v < 0 else v
        gain = 1.0 + nominal_strength * carrier[j] * (2 * <int>bits[k] - 1)
        value = spectrum[row, u] * gain
        spectrum[row, u] = value
        spectrum[512 - row, 512 - u] = value.real - value.imag * 1j
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _extract_core(cnp.ndarray image, bytes key, double strength):
    """
    Извлекает 64 бита ЦВЗ по отношению правдоподобий двух гипотез Вейбулла.

    Args:
        image: проверенная одноканальная матрица uint8 формы (512, 512).
        key: проверенный 32-байтовый ключ перестановки и несущей.
        strength: согласованная сила tau из результата встраивания.

    Returns:
        dict: биты int8, LLR и оценки гипотез float64, параметры Вейбулла и диагностика; нулевой LLR даёт бит 0.
    """
    coordinates, zone_array = dft_layout()
    spectrum = np.fft.fft2(image, norm="backward")
    cdef cnp.ndarray magnitudes = np.ascontiguousarray(
        np.abs(spectrum[coordinates[:, 0] % 512, coordinates[:, 1]]))
    parameters_array, normalized_array = dft_fit_weibull(magnitudes, zone_array)
    permutation_array, carrier_array = dft_key_material(key)
    cdef cnp.ndarray bits_array = np.empty(64, dtype=np.int8)
    cdef cnp.ndarray llr_array = np.empty(64, dtype=np.float64)
    cdef cnp.ndarray scores_array = np.empty((64, 2), dtype=np.float64)
    cdef const double[:, ::1] parameters = parameters_array
    cdef const double[::1] normalized = normalized_array
    cdef const cnp.intp_t[::1] zones = zone_array
    cdef const cnp.intp_t[::1] permutation = permutation_array
    cdef const double[::1] carrier = carrier_array
    cdef cnp.int8_t[::1] bits = bits_array
    cdef double[::1] llr = llr_array
    cdef double[:, ::1] scores = scores_array
    cdef int j, i, k, r
    cdef double a, beta, z, q
    cdef bint valid = True
    with nogil:
        for k in range(64):
            scores[k, 0] = 0.0
            scores[k, 1] = 0.0
        for j in range(18960):
            i = permutation[j]
            r = zones[i]
            k = j // 296
            if k > 63:
                k = 63
            beta, z, q = parameters[r, 1], normalized[i], strength * carrier[j]
            a = -beta * log1p(-q)
            scores[k, 0] += a - z * expm1(a)
            a = -beta * log1p(q)
            scores[k, 1] += a - z * expm1(a)
        for k in range(64):
            llr[k] = scores[k, 1] - scores[k, 0]
            if not isfinite(llr[k]) or not isfinite(scores[k, 0]) or not isfinite(scores[k, 1]):
                valid = False
                break
            bits[k] = 1 if llr[k] > 0.0 else 0
    if not valid:
        raise ArithmeticError("NUMERIC_ERROR: nonfinite DFT likelihood scores")
    result = {"status": "OK", "profile": "barni-dft-psnr-v02", "bits": bits_array.copy(),
              "llr": llr_array.copy(), "scores": scores_array.copy(),
              "weibull_parameters": parameters_array, "strength": strength,
              "zero_magnitudes": int(np.count_nonzero(magnitudes == 0.0)),
              "shape_bound_hits": int(np.count_nonzero(
                  (parameters_array[:, 1] == 0.2) | (parameters_array[:, 1] == 10.0)))}
    return result


class DFT(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        defaults = {"input_image": None, "watermark_bits": None, "key": None}
        unknown = set(args) - defaults.keys()
        if unknown:
            raise TypeError(f"Unknown dft embedding parameters: {sorted(unknown)}")
        args = {**defaults, **args}
        cdef cnp.ndarray image = dft_validate_image(args["input_image"])
        cdef cnp.ndarray watermark = dft_validate_bits(args["watermark_bits"])
        cdef bytes key = dft_validate_key(args["key"])
        cdef double strength, target_psnr = 45.0
        cdef double target_sse = 65025.0 * 262144.0 * 10.0 ** (-target_psnr / 10.0)
        cdef double low = 0.0, high = 1.0, mid
        cdef int iteration
        coordinates, zone_array = dft_layout()
        permutation_array, carrier_array = dft_key_material(key)
        cdef cnp.ndarray spectrum_array = np.ascontiguousarray(np.fft.fft2(image, norm="backward"))
        cdef cnp.ndarray mask_array = np.empty((512, 512), dtype=np.float64)
        cdef cnp.ndarray result_array = np.empty((512, 512), dtype=np.uint8)
        cdef const cnp.uint8_t[:, ::1] original = image
        cdef const cnp.uint8_t[::1] bits = watermark
        cdef cnp.complex128_t[:, ::1] spectrum = spectrum_array
        cdef const cnp.intp_t[:, ::1] coords = coordinates
        cdef const cnp.intp_t[::1] zones = zone_array
        cdef const cnp.intp_t[::1] permutation = permutation_array
        cdef const double[::1] carrier = carrier_array
        cdef double[:, ::1] mask = mask_array
        cdef cnp.uint8_t[:, ::1] result = result_array
        cdef double mean_mask = 1.0, nominal_strength, sse, psnr, imaginary_error
        cdef int clipped = 0
        cdef bint valid
        with nogil:
            valid = dft_build_mask(original, mask)
        if not valid:
            raise ValueError("UNSUITABLE_HOST: local variance is zero everywhere")
        mean_mask = float(np.mean(mask_array))
        nominal_strength = 0.95
        with nogil:
            valid = _embed_core(spectrum, coords, zones, permutation, carrier, bits, nominal_strength)
        if not valid:
            raise ValueError("UNSUITABLE_HOST: a DFT region or bit group contains only zero coefficients")
        restored_complex = np.fft.ifft2(spectrum_array, norm="backward")
        imaginary_error = float(np.max(np.abs(restored_complex.imag)))
        if not isfinite(imaginary_error) or imaginary_error > 1e-8:
            raise ArithmeticError("NUMERIC_ERROR: inverse DFT imaginary residual exceeds 1e-8")
        cdef cnp.ndarray restored_array = np.ascontiguousarray(restored_complex.real)
        cdef const double[:, ::1] restored = restored_array
        with nogil:
            sse = dft_render_pixels(original, restored, mask, result, &clipped)
        if sse < target_sse:
            raise ValueError("TARGET_UNREACHABLE: nominal gain 0.95 cannot reach target PSNR")
        with nogil:
            for iteration in range(40):
                mid = (low + high) / 2.0
                sse = dft_render_pixels(original, restored, mask, result, &clipped, mid)
                if sse <= target_sse:
                    low = mid
                else:
                    high = mid
            sse = dft_render_pixels(original, restored, mask, result, &clipped, low)
        nominal_strength = 0.95 * low
        strength = nominal_strength * mean_mask
        psnr = float("inf") if sse == 0.0 else 10.0 * log10(65025.0 * 262144.0 / sse)
        if not isfinite(psnr) or not target_psnr <= psnr <= target_psnr + 0.001:
            raise ValueError("TARGET_UNREACHABLE: uint8 quantization cannot meet 0.001 dB tolerance")
        return {"status": "OK", "profile": "barni-dft-psnr-v02", "image": result_array,
                "strength": strength, "nominal_strength": nominal_strength,
                "mean_mask": mean_mask, "clipped_pixels": clipped, "psnr_db": psnr,
                "max_ifft_imaginary": imaginary_error, "num_bits": 64,
                "mask_floor": 0.5,
                "target_psnr_db": target_psnr,
                "calibration_iterations": 40,
                "numpy_version": np.__version__}

    @staticmethod
    def extraction(**args):
        defaults = {"input_image": None, "key": None, "strength": None, "return_details": False}
        unknown = set(args) - defaults.keys()
        if unknown:
            raise TypeError(f"Unknown dft extraction parameters: {sorted(unknown)}")
        args = {**defaults, **args}
        cdef cnp.ndarray image = dft_validate_image(args["input_image"])
        cdef bytes key = dft_validate_key(args["key"])
        cdef double strength = dft_validate_strength(args["strength"])
        if not isinstance(args["return_details"], (bool, np.bool_)):
            raise TypeError("return_details must be bool")
        result = _extract_core(image, key, strength)
        return result if args["return_details"] else result["bits"]
