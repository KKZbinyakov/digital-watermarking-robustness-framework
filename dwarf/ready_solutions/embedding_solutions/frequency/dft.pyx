"""
Wilson Wai Lun Fung and Akiomi Kunisa
Rotation, Scaling, and Translation-Invariant Multi-Bit Watermarking based on Log-Polar Mapping and Discrete Fourier Transform (2005)
https://cecs.uci.edu/~papers/icme05/defevent/papers/cr1102.pdf
"""

import numpy as np
cimport numpy as cnp

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    dft_validate_rgb_uint8,
    dft_validate_bits,
    dft_rgb_to_y,
    dft_apply_luma_delta,
    dft_ecc_encode,
    dft_ecc_decode_soft_batch,
    dft_interleave_coded_bits,
    dft_deinterleave_soft_matrix,
    dft_manchester_encode,
    dft_coded_antipodal_from_messages,
    dft_inverse_lpm_backward,
    dft_crop_center,
    dft_rt_invariant_from_image,
    dft_rt_magnitude_from_image,
    dft_rho_index_for_omega,
    dft_scale_delta_to_psnr,
)

cnp.import_array()


cdef object _embed_core(object rgb, object message, str ecc_mode, int interleaver_seed,
                        int fb, int redundancy, double omega1, double omega2,
                        int rho_samples, int theta_samples, double omega_min,
                        double omega_max, int pad_factor, object target_psnr):
    """
    Выполняет основной цикл DFT/LPM-встраивания ЦВЗ по схеме статьи.

    Args:
        rgb: проверенное RGB-изображение uint8 формы (H, W, 3).
        message: проверенный одномерный массив uint8 со значениями 0 и 1.
        ecc_mode: режим ECC для формирования кодового слова.
        interleaver_seed: зерно перестановки ECC-битов перед Manchester-кодированием.
        fb: начальный индекс угловой частоты для Manchester-последовательности.
        redundancy: число соседних строк rho с повторением одного кодового слова.
        omega1: нижняя нормированная частота области встраивания.
        omega2: верхняя нормированная частота области встраивания.
        rho_samples: число отсчётов log-polar представления по rho.
        theta_samples: число отсчётов log-polar представления по theta.
        omega_min: минимальная частота полной log-polar сетки.
        omega_max: максимальная частота полной log-polar сетки.
        pad_factor: коэффициент нулевого дополнения перед двумерным DFT.
        target_psnr: целевой PSNR пространственного встраивания или None.

    Returns:
        output_image: RGB-изображение uint8 формы (H, W, 3) со встроенным ЦВЗ.
    """
    y = dft_rgb_to_y(rgb)
    coded = dft_ecc_encode(message, ecc_mode)
    transmitted_coded = dft_interleave_coded_bits(coded, interleaver_seed)
    manchester = dft_manchester_encode(transmitted_coded)
    max_positive_frequency = fb + manchester.size - 1
    if max_positive_frequency >= theta_samples // 2:
        raise ValueError(
            "Angular capacity exceeded: the Manchester band must remain below "
            "the angular Nyquist frequency. Increase theta_samples or shorten the watermark."
        )
    rt_data = dft_rt_invariant_from_image(
        y,
        rho_samples,
        theta_samples,
        omega_min,
        omega_max,
        pad_factor,
    )
    rho1 = dft_rho_index_for_omega(omega1, rho_samples, omega_min, omega_max)
    rho2_limit = dft_rho_index_for_omega(omega2, rho_samples, omega_min, omega_max)
    rho2 = rho1 + redundancy - 1
    if rho2 >= rho_samples:
        raise ValueError("rho_samples cannot fit requested embedding redundancy")
    if rho2 > rho2_limit:
        raise ValueError("omega1..omega2 interval cannot fit requested embedding redundancy")
    watermark_rt_magnitude = np.zeros((rho_samples, theta_samples), dtype=np.float64)
    frequency_indices = fb + np.arange(manchester.size, dtype=np.int64)
    watermark_rt_magnitude[rho1:rho2 + 1, frequency_indices] = manchester[None, :]
    watermark_rt_magnitude[
        rho1:rho2 + 1,
        (-frequency_indices) % theta_samples,
    ] = manchester[None, :]
    watermark_rt_complex = watermark_rt_magnitude * np.exp(1j * rt_data["rt_phase"])
    watermark_lpm_complex = np.fft.ifft(watermark_rt_complex, axis=1)
    watermark_lpm_signed = watermark_lpm_complex.real
    watermark_lpm = 0.5 * (watermark_lpm_signed + np.abs(watermark_lpm_signed))
    watermark_cartesian_magnitude = dft_inverse_lpm_backward(
        watermark_lpm,
        rt_data["F"].shape,
        omega_min,
        omega_max,
    )
    watermark_cartesian = watermark_cartesian_magnitude * np.exp(1j * rt_data["phase_F"])
    watermark_padded = np.fft.ifft2(np.fft.ifftshift(watermark_cartesian)).real
    watermark_spatial = dft_crop_center(watermark_padded, (y.shape[0], y.shape[1]))
    watermark_spatial = watermark_spatial - float(np.mean(watermark_spatial))
    watermark_spatial = dft_scale_delta_to_psnr(y, watermark_spatial, target_psnr)
    watermarked_y = y + watermark_spatial
    return dft_apply_luma_delta(rgb, y, watermarked_y)


cdef object _extract_core(object rgb, int message_length, str ecc_mode,
                          int interleaver_seed, int fb, int redundancy,
                          int rho_samples, int theta_samples,
                          double omega_min, double omega_max, int pad_factor,
                          double embedding_omega1, double scale_min,
                          double scale_max, double rho_prior_strength,
                          double tau, double mu, bint reject_invalid,
                          bint center_soft, str candidate_mode,
                          bint return_diagnostics):
    """
    Выполняет слепое извлечение ЦВЗ из DFT/LPM-представления.

    Args:
        rgb: проверенное RGB-изображение uint8 формы (H, W, 3).
        message_length: длина исходного сообщения в битах.
        ecc_mode: режим ECC, совпадающий с режимом встраивания.
        interleaver_seed: зерно перестановки ECC-битов, совпадающее с встраиванием.
        fb: начальный индекс угловой частоты Manchester-последовательности.
        redundancy: число соседних строк rho, использованных при встраивании.
        rho_samples: число отсчётов log-polar представления по rho.
        theta_samples: число отсчётов log-polar представления по theta.
        omega_min: минимальная частота полной log-polar сетки.
        omega_max: максимальная частота полной log-polar сетки.
        pad_factor: коэффициент нулевого дополнения перед двумерным DFT.
        embedding_omega1: нижняя частота исходной полосы встраивания.
        scale_min: минимальный коэффициент масштаба, учитываемый поиском по rho.
        scale_max: максимальный коэффициент масштаба, учитываемый поиском по rho.
        rho_prior_strength: слабый штраф за удаление окна от позиции без масштабирования.
        tau: минимальная нормированная корреляция допустимого кандидата.
        mu: порог итогового взвешенного решения для режимов построчного голосования.
        reject_invalid: отклонять ли изображение при отсутствии однозначного решения.
        center_soft: удалять ли общий аддитивный сдвиг мягких Manchester-оценок по строке rho.
        candidate_mode: режим выбора кандидатов paper_all, redundancy_window или pooled_window.
        return_diagnostics: возвращать ли вместе с сообщением диагностические характеристики детектора.

    Returns:
        result: массив int8 с извлечёнными битами либо пара из массива и словаря диагностики.
    """
    y = dft_rgb_to_y(rgb)
    dummy = np.zeros(message_length, dtype=np.uint8)
    coded_length = dft_ecc_encode(dummy, ecc_mode).size
    max_positive_frequency = fb + 2 * coded_length - 1
    if max_positive_frequency >= theta_samples // 2:
        raise ValueError(
            "Extraction geometry reaches the angular Nyquist frequency. "
            "Increase theta_samples or shorten the watermark."
        )
    rt_magnitude = dft_rt_magnitude_from_image(
        y,
        rho_samples,
        theta_samples,
        omega_min,
        omega_max,
        pad_factor,
    )
    first = rt_magnitude[:, fb:fb + 2 * coded_length:2]
    second = rt_magnitude[:, fb + 1:fb + 2 * coded_length:2]
    transmitted_soft = np.ascontiguousarray(first - second, dtype=np.float64)
    soft_matrix = dft_deinterleave_soft_matrix(transmitted_soft, interleaver_seed)
    if center_soft:
        soft_matrix = np.ascontiguousarray(
            soft_matrix - np.mean(soft_matrix, axis=1, keepdims=True),
            dtype=np.float64,
        )
    candidate_bits = dft_ecc_decode_soft_batch(
        soft_matrix,
        message_length,
        ecc_mode,
    )
    reencoded_matrix = dft_coded_antipodal_from_messages(
        candidate_bits,
        ecc_mode,
    )
    soft_norms = np.linalg.norm(soft_matrix, axis=1)
    code_norms = np.linalg.norm(reencoded_matrix, axis=1)
    denominators = soft_norms * code_norms
    numerators = np.einsum("ij,ij->i", soft_matrix, reencoded_matrix)
    candidate_corr = np.full(rho_samples, -1.0, dtype=np.float64)
    valid_norms = denominators > 1e-24
    candidate_corr[valid_norms] = numerators[valid_norms] / denominators[valid_norms]
    best_start = -1
    best_window_score = float("nan")
    best_window_corr = float("nan")
    estimated_scale = float("nan")
    search_start_min = -1
    search_start_max = -1
    pooled_out = None
    selected = np.empty(0, dtype=np.int64)
    if candidate_mode == "paper_all":
        selected = np.flatnonzero(candidate_corr > tau)
    elif candidate_mode == "redundancy_window":
        if redundancy < 1 or redundancy > rho_samples:
            raise ValueError("redundancy must lie in [1, rho_samples]")
        window_scores = np.convolve(
            np.maximum(candidate_corr - tau, 0.0),
            np.ones(redundancy, dtype=np.float64),
            mode="valid",
        )
        best_start = int(np.argmax(window_scores))
        best_window_score = float(window_scores[best_start])
        local = np.arange(best_start, best_start + redundancy, dtype=np.int64)
        selected = local[candidate_corr[local] > tau]
    elif candidate_mode == "pooled_window":
        if redundancy < 1 or redundancy > rho_samples:
            raise ValueError("redundancy must lie in [1, rho_samples]")
        row_norms = np.linalg.norm(soft_matrix, axis=1, keepdims=True)
        normalized_soft = np.divide(
            soft_matrix,
            np.maximum(row_norms, 1e-24),
        )
        prefix = np.vstack(
            (
                np.zeros((1, coded_length), dtype=np.float64),
                np.cumsum(normalized_soft, axis=0),
            )
        )
        pooled_soft = np.ascontiguousarray(
            prefix[redundancy:] - prefix[:-redundancy],
            dtype=np.float64,
        )
        pooled_bits = dft_ecc_decode_soft_batch(
            pooled_soft,
            message_length,
            ecc_mode,
        )
        pooled_codewords = dft_coded_antipodal_from_messages(
            pooled_bits,
            ecc_mode,
        )
        pooled_norms = np.linalg.norm(pooled_soft, axis=1)
        pooled_code_norms = np.linalg.norm(pooled_codewords, axis=1)
        pooled_denominators = pooled_norms * pooled_code_norms
        pooled_numerators = np.einsum("ij,ij->i", pooled_soft, pooled_codewords)
        pooled_corr = np.full(pooled_soft.shape[0], -1.0, dtype=np.float64)
        pooled_valid = pooled_denominators > 1e-24
        pooled_corr[pooled_valid] = (
            pooled_numerators[pooled_valid] / pooled_denominators[pooled_valid]
        )
        reference_rho = dft_rho_index_for_omega(
            embedding_omega1,
            rho_samples,
            omega_min,
            omega_max,
        )
        log_span = float(np.log(omega_max) - np.log(omega_min))
        lower_shift = -float(np.log(scale_max)) / log_span * (rho_samples - 1)
        upper_shift = -float(np.log(scale_min)) / log_span * (rho_samples - 1)
        search_start_min = max(0, int(np.floor(reference_rho + lower_shift)))
        search_start_max = min(
            pooled_corr.size - 1,
            int(np.ceil(reference_rho + upper_shift)),
        )
        if search_start_min > search_start_max:
            raise ValueError("Scale search interval does not intersect the available rho windows")
        starts = np.arange(pooled_corr.size, dtype=np.int64)
        window_scores = np.full(pooled_corr.size, -np.inf, dtype=np.float64)
        valid_search = (starts >= search_start_min) & (starts <= search_start_max)
        window_scores[valid_search] = (
            pooled_corr[valid_search]
            - rho_prior_strength * np.abs(starts[valid_search] - reference_rho)
        )
        best_start = int(np.argmax(window_scores))
        best_window_score = float(window_scores[best_start])
        best_window_corr = float(pooled_corr[best_start])
        estimated_scale = float(
            np.exp(
                -(best_start - reference_rho)
                * log_span
                / max(rho_samples - 1, 1)
            )
        )
        selected = np.arange(best_start, best_start + redundancy, dtype=np.int64)
        if best_window_corr > tau:
            pooled_out = np.ascontiguousarray(
                pooled_bits[best_start],
                dtype=np.int8,
            )
    else:
        raise ValueError(
            "candidate_mode must be 'paper_all', 'redundancy_window' or 'pooled_window'"
        )
    peak_rho = int(np.argmax(candidate_corr))
    diagnostics = {
        "peak_rho": peak_rho,
        "peak_corr": float(candidate_corr[peak_rho]),
        "rho_above_tau": int(np.count_nonzero(candidate_corr > tau)),
        "selected_count": int(selected.size),
        "best_window_start": int(best_start),
        "best_window_end": int(best_start + redundancy - 1) if best_start >= 0 else -1,
        "best_window_score": float(best_window_score),
        "best_window_corr": float(best_window_corr),
        "mean_selected_corr": float(np.mean(candidate_corr[selected])) if selected.size else float("nan"),
        "estimated_scale": float(estimated_scale),
        "search_start_min": int(search_start_min),
        "search_start_max": int(search_start_max),
        "candidate_mode": candidate_mode,
        "tau": float(tau),
    }
    if candidate_mode == "pooled_window":
        if pooled_out is None:
            diagnostics["invalid_bits"] = int(message_length)
            if reject_invalid:
                raise ValueError(
                    "DFT watermark invalid: no pooled rho window passed correlation threshold"
                )
            out = np.full(message_length, -1, dtype=np.int8)
            if return_diagnostics:
                return np.ascontiguousarray(out), diagnostics
            return np.ascontiguousarray(out)
        diagnostics["invalid_bits"] = 0
        if return_diagnostics:
            return pooled_out, diagnostics
        return pooled_out
    if selected.size == 0:
        if reject_invalid:
            raise ValueError("DFT watermark invalid: no rho candidate passed correlation threshold")
        out = np.full(message_length, -1, dtype=np.int8)
        if return_diagnostics:
            return np.ascontiguousarray(out), diagnostics
        return np.ascontiguousarray(out)
    decoded_matrix = candidate_bits[selected]
    weights = candidate_corr[selected]
    antipodal = np.where(decoded_matrix > 0, 1.0, -1.0)
    weighted = (weights[:, None] * antipodal).sum(axis=0)
    out = np.empty(message_length, dtype=np.int8)
    out[weighted > mu] = 1
    out[weighted < -mu] = 0
    invalid = np.abs(weighted) <= mu
    out[invalid] = -1
    diagnostics["invalid_bits"] = int(np.count_nonzero(invalid))
    diagnostics["weighted_min_abs"] = float(np.min(np.abs(weighted)))
    diagnostics["weighted_mean_abs"] = float(np.mean(np.abs(weighted)))
    if reject_invalid and np.any(invalid):
        raise ValueError(
            f"DFT watermark invalid: {int(invalid.sum())} bit(s) are undecidable under Eq. (9)"
        )
    out = np.ascontiguousarray(out)
    if return_diagnostics:
        return out, diagnostics
    return out


class DFT(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        defaults = {
            "input_image": None,
            "watermark_bits": None,
            "ecc_mode": "conv_r13_m6",
            "interleaver_seed": 2005,
            "fb": 32,
            "redundancy": 13,
            "omega1": 0.33,
            "omega2": 0.43,
            "rho_samples": 512,
            "theta_samples": 1024,
            "omega_min": 3.126841665315045e-5,
            "omega_max": 1.0320544788256814,
            "pad_factor": 2,
            "target_psnr": 43.2,
        }
        params = {**defaults, **args}
        if params["input_image"] is None or params["watermark_bits"] is None:
            raise ValueError("input_image and watermark_bits are required")
        rgb = dft_validate_rgb_uint8(params["input_image"])
        message = dft_validate_bits(params["watermark_bits"])
        fb = int(params["fb"])
        redundancy = int(params["redundancy"])
        rho_samples = int(params["rho_samples"])
        theta_samples = int(params["theta_samples"])
        omega1 = float(params["omega1"])
        omega2 = float(params["omega2"])
        omega_min = float(params["omega_min"])
        omega_max = float(params["omega_max"])
        pad_factor = int(params["pad_factor"])
        ecc_mode = str(params["ecc_mode"])
        interleaver_seed = int(params["interleaver_seed"])
        if fb <= 0:
            raise ValueError("fb must be a positive integer")
        if redundancy < 1:
            raise ValueError("redundancy must be >= 1")
        if rho_samples < 2:
            raise ValueError("rho_samples must be >= 2")
        if theta_samples < 2:
            raise ValueError("theta_samples must be >= 2")
        if not (0.0 < omega_min < omega1 < omega2 < omega_max):
            raise ValueError("Need 0 < omega_min < omega1 < omega2 < omega_max")
        return _embed_core(
            rgb,
            message,
            ecc_mode,
            interleaver_seed,
            fb,
            redundancy,
            omega1,
            omega2,
            rho_samples,
            theta_samples,
            omega_min,
            omega_max,
            pad_factor,
            params["target_psnr"],
        )

    @staticmethod
    def extraction(**args):
        defaults = {
            "input_image": None,
            "num_bits": 0,
            "ecc_mode": "conv_r13_m6",
            "interleaver_seed": 2005,
            "fb": 32,
            "redundancy": 13,
            "rho_samples": 512,
            "theta_samples": 1024,
            "omega_min": 3.126841665315045e-5,
            "omega_max": 1.0320544788256814,
            "pad_factor": 2,
            "embedding_omega1": 0.33,
            "scale_min": 0.4,
            "scale_max": 2.0,
            "rho_prior_strength": 0.001,
            "tau": 0.53,
            "mu": 0.0,
            "reject_invalid": True,
            "center_soft": True,
            "candidate_mode": "pooled_window",
            "return_diagnostics": False,
        }
        params = {**defaults, **args}
        if params["input_image"] is None:
            raise ValueError("input_image is required")
        message_length = int(params["num_bits"])
        if message_length <= 0:
            raise ValueError("num_bits must be > 0")
        rgb = dft_validate_rgb_uint8(params["input_image"])
        fb = int(params["fb"])
        redundancy = int(params["redundancy"])
        rho_samples = int(params["rho_samples"])
        theta_samples = int(params["theta_samples"])
        omega_min = float(params["omega_min"])
        omega_max = float(params["omega_max"])
        pad_factor = int(params["pad_factor"])
        ecc_mode = str(params["ecc_mode"])
        interleaver_seed = int(params["interleaver_seed"])
        embedding_omega1 = float(params["embedding_omega1"])
        scale_min = float(params["scale_min"])
        scale_max = float(params["scale_max"])
        rho_prior_strength = float(params["rho_prior_strength"])
        tau = float(params["tau"])
        mu = float(params["mu"])
        reject_invalid = bool(params["reject_invalid"])
        center_soft = bool(params["center_soft"])
        candidate_mode = str(params["candidate_mode"])
        return_diagnostics = bool(params["return_diagnostics"])
        if fb <= 0:
            raise ValueError("fb must be a positive integer")
        if redundancy < 1:
            raise ValueError("redundancy must be >= 1")
        if rho_samples < 2:
            raise ValueError("rho_samples must be >= 2")
        if theta_samples < 2:
            raise ValueError("theta_samples must be >= 2")
        if not (0.0 < omega_min < embedding_omega1 < omega_max):
            raise ValueError("Need 0 < omega_min < embedding_omega1 < omega_max")
        if not (0.0 < scale_min <= scale_max):
            raise ValueError("Need 0 < scale_min <= scale_max")
        if rho_prior_strength < 0.0:
            raise ValueError("rho_prior_strength must be >= 0")
        if not (-1.0 <= tau <= 1.0):
            raise ValueError("tau must lie in [-1, 1]")
        if mu < 0.0:
            raise ValueError("mu must be >= 0")
        return _extract_core(
            rgb,
            message_length,
            ecc_mode,
            interleaver_seed,
            fb,
            redundancy,
            rho_samples,
            theta_samples,
            omega_min,
            omega_max,
            pad_factor,
            embedding_omega1,
            scale_min,
            scale_max,
            rho_prior_strength,
            tau,
            mu,
            reject_invalid,
            center_soft,
            candidate_mode,
            return_diagnostics,
        )
