"""Contourlet/SVD/QIM watermarking profile from Wu et al. (2023).

The implementation follows the algorithm and equations in:
S. Wu, Z. Wu, M. Huang, and D. Shen, "A Robust Image Watermarking
Scheme Based on Image Normalization and Contourlet Transform",
IEEE Access, vol. 11, pp. 105810-105818, 2023,
https://doi.org/10.1109/ACCESS.2023.3319669.

Only the low-pass branch of the one-scale contourlet decomposition is
material to this scheme.  Directional subbands are left unchanged, so they
do not have to be constructed explicitly during embedding.
"""

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
# The paper fixes Delta_0=48 but does not state the coefficient-unit
# convention around its MATLAB contourlet implementation.  This paired
# analysis/synthesis scale leaves perfect reconstruction unchanged and is the
# reproducibility profile's explicit calibration to the reported ~40 dB PSNR.
_LOWPASS_GAIN = 5.0


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


def _lowpass_analysis(image: np.ndarray) -> np.ndarray:
    """One-scale separable pyramid analysis used by the low-pass contourlet branch."""
    filtered = convolve1d(np.asarray(image, dtype=np.float64), _ANALYSIS_TAPS, axis=0, mode="wrap")
    filtered = convolve1d(filtered, _ANALYSIS_TAPS, axis=1, mode="wrap")
    return np.ascontiguousarray(filtered[::2, ::2] * _LOWPASS_GAIN)


def _lowpass_synthesis_delta(coarse_delta: np.ndarray) -> np.ndarray:
    """Maps a low-pass coefficient change back while preserving all high-pass bands."""
    coarse_delta = np.asarray(coarse_delta, dtype=np.float64)
    expanded = np.zeros((2 * coarse_delta.shape[0], 2 * coarse_delta.shape[1]), dtype=np.float64)
    expanded[::2, ::2] = coarse_delta / _LOWPASS_GAIN
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


def _embed_lowpass(lowpass: np.ndarray, bits: np.ndarray, delta0: float, block_size: int) -> np.ndarray:
    result = np.array(lowpass, dtype=np.float64, copy=True)
    carriers = _blocks(result, block_size)
    flat = np.ascontiguousarray(carriers).reshape(-1, block_size, block_size)
    if bits.size > flat.shape[0]:
        raise ValueError(f"Not enough payload capacity: need {bits.size} bits, available {flat.shape[0]}")

    selected = flat[: bits.size]
    # Delta is content-adaptive and extraction recomputes it from the marked
    # block.  Re-evaluating the published rule reaches the same fixed point
    # that the decoder observes and avoids parity changes caused solely by the
    # energy modification of S(1,1).
    for iteration in range(6):
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


def _arnold(bits: np.ndarray, iterations: int, inverse: bool = False) -> np.ndarray:
    if iterations == 0:
        return np.ascontiguousarray(bits)
    side = int(round(bits.size**0.5))
    if side * side != bits.size:
        raise ValueError("Arnold scrambling requires a square binary watermark")

    source = np.asarray(bits).reshape(side, side)
    yy, xx = np.indices((side, side))
    if inverse:
        target_x = (2 * xx - yy) % side
        target_y = (-xx + yy) % side
    else:
        target_x = (xx + yy) % side
        target_y = (xx + 2 * yy) % side

    result = source
    for _ in range(iterations):
        transformed = np.empty_like(result)
        transformed[target_y, target_x] = result[yy, xx]
        result = transformed
    return np.ascontiguousarray(result.reshape(-1))


def _moment_transform(image: np.ndarray, target_std: float, output_shape: tuple[int, int]) -> np.ndarray:
    """Build an affine canonical frame from second-, third-, and fifth-order moments."""
    weights = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
    yy, xx = np.indices(weights.shape, dtype=np.float64)
    mass = weights.sum()
    if mass <= np.finfo(np.float64).eps:
        raise ValueError("image has zero intensity mass and cannot be normalized")

    centroid = np.array([(weights * xx).sum() / mass, (weights * yy).sum() / mass])
    dx = xx - centroid[0]
    dy = yy - centroid[1]
    covariance = (
        np.array(
            [
                [(weights * dx * dx).sum(), (weights * dx * dy).sum()],
                [(weights * dx * dy).sum(), (weights * dy * dy).sum()],
            ]
        )
        / mass
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[0] <= 1e-9:
        raise ValueError("image moments are degenerate and cannot define an affine normalization")
    order = np.argsort(eigenvalues)[::-1]
    whitening = np.diag(1.0 / np.sqrt(eigenvalues[order])) @ eigenvectors[:, order].T

    whitened_x = whitening[0, 0] * dx + whitening[0, 1] * dy
    whitened_y = whitening[1, 0] * dx + whitening[1, 1] * dy
    radius_squared = whitened_x * whitened_x + whitened_y * whitened_y
    direction = (
        np.array(
            [
                (weights * whitened_x * radius_squared).sum(),
                (weights * whitened_y * radius_squared).sum(),
            ]
        )
        / mass
    )
    if np.linalg.norm(direction) > 1e-8:
        angle = np.arctan2(direction[1], direction[0])
        rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        oriented = rotation @ whitening
    else:
        oriented = whitening

    oriented_y = oriented[1, 0] * dx + oriented[1, 1] * dy
    fifth_moment = (weights * oriented_y**5).sum() / mass
    if fifth_moment < 0.0:
        oriented[1] *= -1.0

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


def _roi_origin(image: np.ndarray, roi_size: int) -> tuple[int, int]:
    # The affine normalization places the full-image moment centroid at the
    # canonical centre.  Using that invariant point directly is considerably
    # more stable than rounding every intermediate point of the iterative
    # construction; the latter is defined in continuous coordinates in the
    # paper but becomes content-dependent on a discrete raster.
    weights = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
    yy, xx = np.indices(weights.shape, dtype=np.float64)
    mass = weights.sum()
    if mass <= np.finfo(np.float64).eps:
        center_x = (image.shape[1] - 1.0) / 2.0
        center_y = (image.shape[0] - 1.0) / 2.0
    else:
        center_x = float((weights * xx).sum() / mass)
        center_y = float((weights * yy).sum() / mass)
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
    canonical_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    luma = _to_luma(image)
    if not normalize:
        return luma.copy(), np.eye(3, dtype=np.float64)
    output_shape = (canonical_size, canonical_size)
    transform = _moment_transform(luma, target_std, output_shape)
    return _warp_forward(luma, transform, output_shape), transform


def _profile_geometry(roi_size: int, block_size: int, canonical_size: int) -> None:
    if roi_size % (2 * block_size):
        raise ValueError("roi_size must be divisible by 2 * block_size")
    if canonical_size < roi_size:
        raise ValueError(f"roi_size={roi_size} does not fit canonical_size={canonical_size}")
    if canonical_size % 2:
        raise ValueError("canonical_size must be even")


class Contourlet_Wu2023(Ready_Quantization_Based_Embeddings):
    """Blind contourlet/SVD adaptive-QIM watermarking from Wu et al. (2023)."""

    @staticmethod
    def embedding(**args):
        defaults = {
            "input_image": None,
            "watermark_bits": None,
            "delta0": 48.0,
            "roi_size": 256,
            "block_size": 4,
            "arnold_iterations": 0,
            "normalize": True,
            "normalization_std": 256.0,
            "canonical_size": 512,
            "compensation_iterations": 7,
        }
        # The pipeline supplies its complete runtime artifact mapping to every
        # solution.  Read only this implementation's declared parameters.
        args = {**defaults, **{name: args[name] for name in defaults.keys() & args.keys()}}

        image = _validate_image(args["input_image"])
        bits = _validate_bits(args["watermark_bits"])
        delta0 = _positive_float("delta0", args["delta0"])
        roi_size = _integer("roi_size", args["roi_size"], 8)
        block_size = _integer("block_size", args["block_size"], 1)
        arnold_iterations = _integer("arnold_iterations", args["arnold_iterations"])
        normalization_std = _positive_float("normalization_std", args["normalization_std"])
        canonical_size = _integer("canonical_size", args["canonical_size"], roi_size)
        compensation_iterations = _integer("compensation_iterations", args["compensation_iterations"])
        normalize = _boolean("normalize", args["normalize"])
        _profile_geometry(roi_size, block_size, canonical_size)

        payload = _arnold(bits, arnold_iterations)
        normalized, transform = _normalized_luma(image, normalize, normalization_std, canonical_size)
        top, left = _roi_origin(normalized, roi_size) if normalize else _centered_roi_origin(normalized, roi_size)
        roi = normalized[top : top + roi_size, left : left + roi_size]
        lowpass = _lowpass_analysis(roi)
        modified_lowpass = _embed_lowpass(lowpass, payload, delta0, block_size)
        target = normalized.copy()
        target[top : top + roi_size, left : left + roi_size] = roi + _lowpass_synthesis_delta(
            modified_lowpass - lowpass
        )

        result = image.astype(np.float64)
        passes = compensation_iterations + 1 if normalize else 1
        for _ in range(passes):
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
            "block_size": 4,
            "arnold_iterations": 0,
            "normalize": True,
            "normalization_std": 256.0,
            "canonical_size": 512,
        }
        args = {**defaults, **{name: args[name] for name in defaults.keys() & args.keys()}}

        image = _validate_image(args["input_image"])
        num_bits = _integer("num_bits", args["num_bits"], 1)
        delta0 = _positive_float("delta0", args["delta0"])
        roi_size = _integer("roi_size", args["roi_size"], 8)
        block_size = _integer("block_size", args["block_size"], 1)
        arnold_iterations = _integer("arnold_iterations", args["arnold_iterations"])
        normalization_std = _positive_float("normalization_std", args["normalization_std"])
        canonical_size = _integer("canonical_size", args["canonical_size"], roi_size)
        normalize = _boolean("normalize", args["normalize"])
        _profile_geometry(roi_size, block_size, canonical_size)

        normalized, _ = _normalized_luma(image, normalize, normalization_std, canonical_size)
        top, left = _roi_origin(normalized, roi_size) if normalize else _centered_roi_origin(normalized, roi_size)
        roi = normalized[top : top + roi_size, left : left + roi_size]
        extracted = _extract_lowpass(_lowpass_analysis(roi), num_bits, delta0, block_size)
        return _arnold(extracted, arnold_iterations, inverse=True).astype(np.int8, copy=False)
