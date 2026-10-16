"""Contourlet/SVD/QIM watermarking profile from Wu et al. (2023).

The implementation follows the algorithm and equations in:
S. Wu, Z. Wu, M. Huang, and D. Shen, "A Robust Image Watermarking
Scheme Based on Image Normalization and Contourlet Transform",
IEEE Access, vol. 11, pp. 105810-105818, 2023,
https://doi.org/10.1109/ACCESS.2023.3319669.

Only the low-pass branch of the one-scale contourlet decomposition is
material to this scheme.  Directional subbands are left unchanged, so they
do not have to be constructed explicitly during embedding.  Parameters that
the paper treats as secret or does not report are exposed explicitly so a
numerical reproduction never depends on hidden calibration constants.
"""

import math
import operator

import numpy as np
from scipy.ndimage import affine_transform, convolve1d

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Quantization_Based_Embeddings

_ANALYSIS_TAPS = np.array(
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
_SYNTHESIS_TAPS = np.array(
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
_SWAP_XY = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


def _validate_image(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"input_image must have shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"input_image must have dtype uint8, got {image.dtype}")
    if image.shape[0] < 16 or image.shape[1] < 16:
        raise ValueError("input_image is too small for contourlet watermarking")
    return image


def _validate_bits(value: np.ndarray) -> np.ndarray:
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


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {result}")
    return result


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _to_luma(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float64, copy=False)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _lowpass_analysis(image: np.ndarray, coefficient_scale: float = 1.0) -> np.ndarray:
    """One-scale separable pyramid analysis used by the low-pass contourlet branch."""
    filtered = convolve1d(np.asarray(image, dtype=np.float64), _ANALYSIS_TAPS, axis=0, mode="wrap")
    filtered = convolve1d(filtered, _ANALYSIS_TAPS, axis=1, mode="wrap")
    return np.ascontiguousarray(filtered[::2, ::2] * coefficient_scale)


def _lowpass_synthesis_delta(coarse_delta: np.ndarray, coefficient_scale: float = 1.0) -> np.ndarray:
    """Maps a low-pass coefficient change back while preserving all high-pass bands."""
    coarse_delta = np.asarray(coarse_delta, dtype=np.float64)
    expanded = np.zeros((2 * coarse_delta.shape[0], 2 * coarse_delta.shape[1]), dtype=np.float64)
    expanded[::2, ::2] = coarse_delta / coefficient_scale
    expanded = convolve1d(expanded, _SYNTHESIS_TAPS, axis=0, mode="wrap")
    return convolve1d(expanded, _SYNTHESIS_TAPS, axis=1, mode="wrap")


def _blocks(image: np.ndarray, block_size: int) -> np.ndarray:
    rows = image.shape[0] // block_size
    columns = image.shape[1] // block_size
    active = image[: rows * block_size, : columns * block_size]
    return active.reshape(rows, block_size, columns, block_size).transpose(0, 2, 1, 3)


def _adaptive_steps(blocks: np.ndarray, delta0: float) -> np.ndarray:
    """Equation (7): Delta=log2((E_block+1000)/1000)*Delta_0."""
    energy = np.mean(np.square(blocks, dtype=np.float64), axis=(-2, -1))
    # Exact black blocks make the published expression zero and therefore
    # cannot be quantized.  A one-level energy floor is the limiting digital
    # case and only affects blocks that carry no image information at all.
    energy = np.maximum(energy, 1.0)
    return np.log2((energy + 1000.0) / 1000.0) * delta0


def _embed_lowpass(
    lowpass: np.ndarray,
    bits: np.ndarray,
    delta0: float,
    block_size: int,
    max_iterations: int = 6,
) -> np.ndarray:
    result = np.array(lowpass, dtype=np.float64, copy=True)
    carriers = _blocks(result, block_size)
    flat = np.ascontiguousarray(carriers).reshape(-1, block_size, block_size)
    if bits.size > flat.shape[0]:
        raise ValueError(f"Not enough payload capacity: need {bits.size} bits, available {flat.shape[0]}")

    selected = flat[: bits.size]
    # Extraction recomputes Delta from the marked block.  Iterate equations
    # (7)-(9) to make that content-adaptive step self-consistent; otherwise a
    # bit can change solely because embedding changed its block energy.
    for iteration in range(max_iterations):
        steps = _adaptive_steps(selected, delta0)
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


def _extract_lowpass(lowpass: np.ndarray, num_bits: int, delta0: float, block_size: int) -> np.ndarray:
    carriers = _blocks(np.asarray(lowpass, dtype=np.float64), block_size)
    flat = carriers.reshape(-1, block_size, block_size)
    if num_bits > flat.shape[0]:
        raise ValueError(f"Not enough payload capacity: need {num_bits} bits, available {flat.shape[0]}")
    selected = flat[:num_bits]
    steps = _adaptive_steps(selected, delta0)
    singular = np.linalg.svd(selected, compute_uv=False)
    quantized = np.ceil(singular[:, 0] / steps).astype(np.int64)
    return np.ascontiguousarray((1 - quantized % 2).astype(np.int8))


def _arnold(
    bits: np.ndarray,
    iterations: int,
    inverse: bool = False,
    matrix: tuple[int, int, int, int] = (1, 1, 1, 2),
) -> np.ndarray:
    if iterations == 0:
        return np.ascontiguousarray(bits)
    side = int(round(bits.size**0.5))
    if side * side != bits.size:
        raise ValueError("Arnold scrambling requires a square binary watermark")

    source = np.asarray(bits).reshape(side, side)
    yy, xx = np.indices((side, side))
    a, b, c, d = matrix
    determinant = a * d - b * c
    if math.gcd(determinant, side) != 1:
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


def _weighted_moment(weights: np.ndarray, x: np.ndarray, y: np.ndarray, p: int, q: int) -> float:
    return float(np.sum(weights * x**p * y**q) / np.sum(weights))


def _moment_transform(image: np.ndarray, target_std: float, output_shape: tuple[int, int]) -> np.ndarray:
    """Build the four-stage affine normalization from equations (1)-(4)."""
    weights = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
    yy, xx = np.indices(weights.shape, dtype=np.float64)
    mass = weights.sum()
    if mass <= np.finfo(np.float64).eps:
        raise ValueError("image has zero intensity mass and cannot be normalized")

    centroid = np.array([(weights * xx).sum() / mass, (weights * yy).sum() / mass])
    dx = xx - centroid[0]
    dy = yy - centroid[1]
    mu02 = _weighted_moment(weights, dx, dy, 0, 2)
    if mu02 <= 1e-12:
        raise ValueError("image moments are degenerate and cannot define an affine normalization")

    # (2) x-shear, beta=-mu11/mu02.
    beta = -_weighted_moment(weights, dx, dy, 1, 1) / mu02
    shear = np.array([[1.0, beta], [0.0, 1.0]], dtype=np.float64)
    sheared_x = dx + beta * dy
    sheared_y = dy

    # (3) independent scaling.  The signs make the two fifth-order moments
    # positive exactly as required by the paper.
    mu20_sheared = _weighted_moment(weights, sheared_x, sheared_y, 2, 0)
    mu02_sheared = _weighted_moment(weights, sheared_x, sheared_y, 0, 2)
    if min(mu20_sheared, mu02_sheared) <= 1e-12:
        raise ValueError("image moments are degenerate and cannot define an affine normalization")
    mu50_sheared = _weighted_moment(weights, sheared_x, sheared_y, 5, 0)
    mu05_sheared = _weighted_moment(weights, sheared_x, sheared_y, 0, 5)
    alpha = (1.0 if mu50_sheared >= 0.0 else -1.0) / np.sqrt(mu20_sheared)
    delta = (1.0 if mu05_sheared >= 0.0 else -1.0) / np.sqrt(mu02_sheared)
    scaling = np.diag([alpha, delta])
    scaled = scaling @ shear
    scaled_x = scaled[0, 0] * dx + scaled[0, 1] * dy
    scaled_y = scaled[1, 0] * dx + scaled[1, 1] * dy

    # (4) rotation, psi=-atan((mu30+mu12)/(mu21+mu03)).
    numerator = _weighted_moment(weights, scaled_x, scaled_y, 3, 0) + _weighted_moment(
        weights, scaled_x, scaled_y, 1, 2
    )
    denominator = _weighted_moment(weights, scaled_x, scaled_y, 2, 1) + _weighted_moment(
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


def _warp_forward(
    image: np.ndarray,
    source_to_normalized: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    inverse = np.linalg.inv(source_to_normalized)
    matrix = _SWAP_XY @ inverse[:2, :2] @ _SWAP_XY
    offset = _SWAP_XY @ inverse[:2, 2]
    return affine_transform(
        np.asarray(image, dtype=np.float64),
        matrix,
        offset=offset,
        output_shape=output_shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _warp_inverse_delta(
    normalized_delta: np.ndarray,
    source_to_normalized: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    matrix = _SWAP_XY @ source_to_normalized[:2, :2] @ _SWAP_XY
    offset = _SWAP_XY @ source_to_normalized[:2, 2]
    return affine_transform(
        np.asarray(normalized_delta, dtype=np.float64),
        matrix,
        offset=offset,
        output_shape=output_shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _invariant_centroid(image: np.ndarray, radius: float, iterations: int = 20) -> tuple[float, float]:
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


def _roi_origin(image: np.ndarray, roi_size: int, centroid_radius: float) -> tuple[int, int]:
    center_x, center_y = _invariant_centroid(image, centroid_radius)
    top = int(round(center_y - (roi_size - 1.0) / 2.0))
    left = int(round(center_x - (roi_size - 1.0) / 2.0))
    top = min(max(top, 0), image.shape[0] - roi_size)
    left = min(max(left, 0), image.shape[1] - roi_size)
    return top, left


def _centered_roi_origin(image: np.ndarray, roi_size: int) -> tuple[int, int]:
    return (image.shape[0] - roi_size) // 2, (image.shape[1] - roi_size) // 2


def _normalized_luma(
    image: np.ndarray,
    normalize: bool,
    target_std: float,
    canonical_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    luma = _to_luma(image)
    if not normalize:
        return luma.copy(), np.eye(3, dtype=np.float64)
    transform = _moment_transform(luma, target_std, canonical_shape)
    return _warp_forward(luma, transform, canonical_shape), transform


def _profile_geometry(roi_size: int, block_size: int, canonical_shape: tuple[int, int]) -> None:
    if roi_size % (2 * block_size):
        raise ValueError("roi_size must be divisible by 2 * block_size")
    if min(canonical_shape) < roi_size:
        raise ValueError(f"roi_size={roi_size} does not fit canonical_shape={canonical_shape}")


class Contourlet_Wu2023(Ready_Quantization_Based_Embeddings):
    """Blind contourlet/SVD adaptive-QIM watermarking from Wu et al. (2023)."""

    @staticmethod
    def embedding(**args):
        defaults = {
            "input_image": None,
            "watermark_bits": None,
            "delta0": 48.0,
            "roi_size": 256,
            "centroid_radius": 128.0,
            "block_size": 4,
            "coefficient_scale": 1.0,
            "quantization_iterations": 6,
            "arnold_iterations": 0,
            "arnold_a": 1,
            "arnold_b": 1,
            "arnold_c": 1,
            "arnold_d": 2,
            "normalize": True,
            "normalization_std": 256.0,
            "canonical_size": None,
            "canonical_height": 512,
            "canonical_width": 512,
            "compensation_iterations": 0,
        }
        # The pipeline supplies its complete runtime artifact mapping to every
        # solution.  Read only this implementation's declared parameters.
        args = {**defaults, **{name: args[name] for name in defaults.keys() & args.keys()}}

        image = _validate_image(args["input_image"])
        bits = _validate_bits(args["watermark_bits"])
        delta0 = _positive_float("delta0", args["delta0"])
        roi_size = _integer("roi_size", args["roi_size"], 8)
        centroid_radius = _positive_float("centroid_radius", args["centroid_radius"])
        block_size = _integer("block_size", args["block_size"], 1)
        coefficient_scale = _positive_float("coefficient_scale", args["coefficient_scale"])
        quantization_iterations = _integer("quantization_iterations", args["quantization_iterations"], 1)
        arnold_iterations = _integer("arnold_iterations", args["arnold_iterations"])
        arnold_matrix = tuple(_integer(f"arnold_{name}", args[f"arnold_{name}"]) for name in "abcd")
        normalization_std = _positive_float("normalization_std", args["normalization_std"])
        if args["canonical_size"] is None:
            canonical_shape = (
                _integer("canonical_height", args["canonical_height"], roi_size),
                _integer("canonical_width", args["canonical_width"], roi_size),
            )
        else:
            canonical_size = _integer("canonical_size", args["canonical_size"], roi_size)
            canonical_shape = (canonical_size, canonical_size)
        compensation_iterations = _integer("compensation_iterations", args["compensation_iterations"])
        normalize = _boolean("normalize", args["normalize"])
        _profile_geometry(roi_size, block_size, canonical_shape)

        payload = _arnold(bits, arnold_iterations, matrix=arnold_matrix)
        normalized, transform = _normalized_luma(image, normalize, normalization_std, canonical_shape)
        top, left = (
            _roi_origin(normalized, roi_size, centroid_radius)
            if normalize
            else _centered_roi_origin(normalized, roi_size)
        )
        roi = normalized[top : top + roi_size, left : left + roi_size]
        lowpass = _lowpass_analysis(roi, coefficient_scale)
        modified_lowpass = _embed_lowpass(
            lowpass,
            payload,
            delta0,
            block_size,
            max_iterations=quantization_iterations,
        )
        target = normalized.copy()
        target[top : top + roi_size, left : left + roi_size] = roi + _lowpass_synthesis_delta(
            modified_lowpass - lowpass,
            coefficient_scale,
        )

        # The pre-distortion compensation cited by the paper inverse-normalizes
        # both normalized images and adds their difference to the host once.
        normalized_delta = target - normalized
        correction = (
            _warp_inverse_delta(normalized_delta, transform, image.shape[:2]) if normalize else normalized_delta
        )
        result = np.clip(image.astype(np.float64) + correction[..., None], 0.0, 255.0)

        # Retained as an explicit non-paper extension for old experiment files.
        for _ in range(compensation_iterations):
            current = np.floor(np.clip(result, 0.0, 255.0) + 0.5).astype(np.uint8)
            current_luma = _to_luma(current)
            current_normalized = _warp_forward(current_luma, transform, normalized.shape) if normalize else current_luma
            residual = target - current_normalized
            correction = _warp_inverse_delta(residual, transform, image.shape[:2]) if normalize else residual
            result += correction[..., None]
            result = np.clip(result, 0.0, 255.0)

        return np.ascontiguousarray(np.floor(result + 0.5).astype(np.uint8))

    @staticmethod
    def extraction(**args):
        defaults = {
            "input_image": None,
            "num_bits": None,
            "delta0": 48.0,
            "roi_size": 256,
            "centroid_radius": 128.0,
            "block_size": 4,
            "coefficient_scale": 1.0,
            "arnold_iterations": 0,
            "arnold_a": 1,
            "arnold_b": 1,
            "arnold_c": 1,
            "arnold_d": 2,
            "normalize": True,
            "normalization_std": 256.0,
            "canonical_size": None,
            "canonical_height": 512,
            "canonical_width": 512,
        }
        args = {**defaults, **{name: args[name] for name in defaults.keys() & args.keys()}}

        image = _validate_image(args["input_image"])
        num_bits = _integer("num_bits", args["num_bits"], 1)
        delta0 = _positive_float("delta0", args["delta0"])
        roi_size = _integer("roi_size", args["roi_size"], 8)
        centroid_radius = _positive_float("centroid_radius", args["centroid_radius"])
        block_size = _integer("block_size", args["block_size"], 1)
        coefficient_scale = _positive_float("coefficient_scale", args["coefficient_scale"])
        arnold_iterations = _integer("arnold_iterations", args["arnold_iterations"])
        arnold_matrix = tuple(_integer(f"arnold_{name}", args[f"arnold_{name}"]) for name in "abcd")
        normalization_std = _positive_float("normalization_std", args["normalization_std"])
        if args["canonical_size"] is None:
            canonical_shape = (
                _integer("canonical_height", args["canonical_height"], roi_size),
                _integer("canonical_width", args["canonical_width"], roi_size),
            )
        else:
            canonical_size = _integer("canonical_size", args["canonical_size"], roi_size)
            canonical_shape = (canonical_size, canonical_size)
        normalize = _boolean("normalize", args["normalize"])
        _profile_geometry(roi_size, block_size, canonical_shape)

        normalized, _ = _normalized_luma(image, normalize, normalization_std, canonical_shape)
        top, left = (
            _roi_origin(normalized, roi_size, centroid_radius)
            if normalize
            else _centered_roi_origin(normalized, roi_size)
        )
        roi = normalized[top : top + roi_size, left : left + roi_size]
        extracted = _extract_lowpass(
            _lowpass_analysis(roi, coefficient_scale),
            num_bits,
            delta0,
            block_size,
        )
        return _arnold(extracted, arnold_iterations, inverse=True, matrix=arnold_matrix).astype(np.int8, copy=False)
