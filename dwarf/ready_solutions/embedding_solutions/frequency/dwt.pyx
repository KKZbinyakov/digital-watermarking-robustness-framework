"""
X. Zhu et al.
Normalized Correlation-based Quantization Modulation for Robust Watermarking (2014)
https://doi.org/10.1109/TMM.2014.2340695
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log10

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    dwt_vector, apply_haar_dwt, dwt_integer,
    dwt_validate_image, dwt_validate_bits, dwt_key_material, dwt_prepare_core,
    dwt_embed_vector, dwt_decode_vector, dwt_render, dwt_calibrate,
)

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint _embed_core(const double[:, ::1] unit, const double[:, ::1] tangent,
                      const double[:, ::1] stats, const cnp.intp_t[::1] permutation,
                      const double[::1] eta, const cnp.uint8_t[::1] bits,
                      int h, double[:, ::1] targets) noexcept nogil:
    """
    Встраивает биты ЦВЗ квантованием нормированной корреляции векторов.

    Args:
        unit: неизменные единичные опорные векторы формы (256, 32).
        tangent: неизменные ортогональные направления формы (256, 32).
        stats: неизменные корреляции, нормы и проекции формы (256, 4).
        permutation: ключевая перестановка 4096 позиций.
        eta: ключевые доли смещения длины 256.
        bits: проверенный непрерывный массив uint8 с 256 битами 0 и 1.
        h: целый параметр шага от 1 до 65536; delta=h/65536.
        targets: выходные разности Y1 и Y2 формы (2, 4096) без перестановки.
    """
    cdef dwt_vector output
    cdef int j, k, offset, band
    cdef double delta = h / 65536.0
    for j in range(256):
        if not dwt_embed_vector(&unit[j, 0], &tangent[j, 0], &stats[j, 0],
                                  bits[j], delta, delta * eta[j], output):
            return False
        band, offset = j // 128, (j % 128) * 32
        for k in range(32):
            targets[band, permutation[offset + k]] = output[k]
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _extract_core(const double[:, ::1] coeffs, const cnp.intp_t[::1] permutation,
                        const double[:, ::1] reference, const double[::1] eta,
                        int h, cnp.int8_t[::1] bits, cnp.uint8_t[::1] erased,
                        cnp.uint8_t[::1] ambiguous, double[::1] margins) noexcept nogil:
    """
    Извлекает биты ЦВЗ по нормированной корреляции полученных векторов.

    Args:
        coeffs: полученная матрица Haar float64 формы (512, 512), три уровня.
        permutation: воспроизведённая по ключу перестановка 4096 позиций.
        reference: воспроизведённые опорные векторы формы (256, 32).
        eta: воспроизведённые доли смещения длины 256.
        h: согласованный целый параметр шага от 1 до 65536.
        bits: выходной непрерывный массив int8 длины 256; стирания дают бит 0.
        erased: выходные флаги нулевой нормы длины 256.
        ambiguous: выходные флаги равных расстояний до кодовых книг длины 256.
        margins: выходные разности расстояний до кодовых книг длины 256.
    """
    cdef dwt_vector x
    cdef int j, k, index, r, c, offset, e, a
    cdef double delta = h / 65536.0, margin
    for j in range(256):
        offset = (j % 128) * 32
        for k in range(32):
            index = <int>permutation[offset + k]
            r, c = index // 64, index % 64
            if j < 128:
                x[k] = coeffs[r + 64, c + 64] - coeffs[r, c + 64]
            else:
                x[k] = coeffs[r + 64, c] - coeffs[r + 64, c + 64]
        bits[j] = dwt_decode_vector(x, &reference[j, 0], delta, delta * eta[j], &e, &a, &margin)
        erased[j], ambiguous[j], margins[j] = e, a, margin


class DWT(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        defaults = {"input_image": None, "watermark_bits": None, "key": 0,
                    "h": None, "storage": "uint8", "return_details": None}
        unknown = set(args) - defaults.keys()
        if unknown:
            raise TypeError(f"Unknown dwt embedding parameters: {sorted(unknown)}")
        args = {**defaults, **args}
        cdef cnp.ndarray image = dwt_validate_image(args["input_image"], True)
        cdef cnp.ndarray watermark = dwt_validate_bits(args["watermark_bits"])
        cdef bint automatic = args["h"] is None
        details = automatic if args["return_details"] is None else args["return_details"]
        if not isinstance(details, (bool, np.bool_)):
            raise TypeError("return_details must be bool or None")
        if automatic and not details:
            raise ValueError("automatic calibration requires return_details=True to retain h")
        if args["storage"] not in ("float64", "uint8"):
            raise ValueError("storage must be 'float64' or 'uint8'")
        cdef int h = 0 if automatic else dwt_integer("h", args["h"], 1, 65536)
        permutation_array, reference_array, eta_array = dwt_key_material(args["key"])
        cdef const cnp.intp_t[::1] permutation = permutation_array
        cdef const double[:, ::1] reference = reference_array
        cdef const double[::1] eta = eta_array
        cdef const cnp.uint8_t[::1] bits = watermark
        cdef cnp.ndarray coefficients = image.copy()
        cdef cnp.ndarray scratch_array = np.empty((512, 512), dtype=np.float64)
        cdef double[:, ::1] coeffs = coefficients
        cdef double[:, ::1] scratch = scratch_array
        cdef cnp.ndarray unit_array = np.empty((256, 32), dtype=np.float64)
        cdef cnp.ndarray tangent_array = np.empty((256, 32), dtype=np.float64)
        cdef cnp.ndarray stats_array = np.empty((256, 4), dtype=np.float64)
        cdef cnp.ndarray flags_array = np.empty(256, dtype=np.uint8)
        cdef cnp.ndarray targets_array = np.empty((2, 4096), dtype=np.float64)
        cdef double[:, ::1] unit = unit_array
        cdef double[:, ::1] tangent = tangent_array
        cdef double[:, ::1] stats = stats_array
        cdef cnp.uint8_t[::1] flags = flags_array
        cdef double[:, ::1] targets = targets_array
        cdef bint valid
        cdef long long expected_sse = -1
        cdef double psnr, mse
        calibration = None
        with nogil:
            apply_haar_dwt(coeffs, scratch, 3)
            dwt_prepare_core(coeffs, permutation, reference, unit, tangent, stats, flags)
        if automatic:
            h, expected_sse, count, full_count = dwt_calibrate(
                image, coefficients, unit, tangent, stats, permutation, eta, bits, targets, _embed_core)
            calibration = {"candidates": 65536, "valid_candidates": count,
                           "full_idwt_fallbacks": full_count, "target_psnr_db": 42.0,
                           "tolerance_db": 0.01, "selection_uses_ber": False}
        with nogil:
            valid = _embed_core(unit, tangent, stats, permutation, eta, bits, h, targets)
        if not valid:
            raise RuntimeError("dwt projection is nonpositive for a nonzero source vector")
        floating, quantized = dwt_render(coefficients, targets)
        mse = float(np.mean((quantized - image) ** 2))
        psnr = float("inf") if mse == 0 else 10.0 * log10(65025.0 / mse)
        if automatic and int(np.sum((quantized - image) ** 2)) != expected_sse:
            raise RuntimeError("dwt calibration SSE differs from full Haar synthesis")
        result = quantized if args["storage"] == "uint8" else floating
        if not details:
            return result
        return {"image": result, "profile": "dwt-256-T78-v1", "h": h,
                "delta": h / 65536.0, "storage": args["storage"], "psnr_db": psnr,
                "numpy_version": np.__version__, "num_bits": 256,
                "zero_vectors": int(np.count_nonzero(flags_array & 1)),
                "parallel_vectors": int(np.count_nonzero(flags_array & 2)),
                "zero_reference_vectors": int(np.count_nonzero(flags_array & 4)),
                "calibration": calibration}

    @staticmethod
    def extraction(**args):
        defaults = {"input_image": None, "key": 0, "h": None,
                    "num_bits": 256, "return_details": False}
        unknown = set(args) - defaults.keys()
        if unknown:
            raise TypeError(f"Unknown dwt extraction parameters: {sorted(unknown)}")
        args = {**defaults, **args}
        cdef cnp.ndarray image = dwt_validate_image(args["input_image"], False)
        cdef int h = dwt_integer("h", args["h"], 1, 65536)
        dwt_integer("num_bits", args["num_bits"], 256, 256)
        if not isinstance(args["return_details"], (bool, np.bool_)):
            raise TypeError("return_details must be bool")
        permutation_array, reference_array, eta_array = dwt_key_material(args["key"])
        cdef const cnp.intp_t[::1] permutation = permutation_array
        cdef const double[:, ::1] reference = reference_array
        cdef const double[::1] eta = eta_array
        cdef cnp.ndarray coefficients = image.copy()
        cdef cnp.ndarray scratch_array = np.empty((512, 512), dtype=np.float64)
        cdef double[:, ::1] coeffs = coefficients
        cdef double[:, ::1] scratch = scratch_array
        cdef cnp.ndarray bits_array = np.empty(256, dtype=np.int8)
        cdef cnp.ndarray erased_array = np.empty(256, dtype=np.uint8)
        cdef cnp.ndarray ambiguous_array = np.empty(256, dtype=np.uint8)
        cdef cnp.ndarray margins_array = np.empty(256, dtype=np.float64)
        cdef cnp.int8_t[::1] bits = bits_array
        cdef cnp.uint8_t[::1] erased = erased_array
        cdef cnp.uint8_t[::1] ambiguous = ambiguous_array
        cdef double[::1] margins = margins_array
        with nogil:
            apply_haar_dwt(coeffs, scratch, 3)
            _extract_core(coeffs, permutation, reference, eta, h, bits, erased, ambiguous, margins)
        if args["return_details"]:
            return {"bits": bits_array, "erased": erased_array.astype(bool),
                    "ambiguous": ambiguous_array.astype(bool), "margins": margins_array}
        return bits_array
