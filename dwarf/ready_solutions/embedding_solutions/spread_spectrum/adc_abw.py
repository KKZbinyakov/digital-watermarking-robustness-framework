"""Blind ADC-ABW spread-spectrum watermarking.

This is a clean-room implementation of the many-to-one profile proposed in:
H. Guo et al., "Spread Spectrum Watermark in DC: A View from the Embedding
Processing", ACM TOMM, 2025, https://doi.org/10.1145/3743139.
"""

from functools import lru_cache
from numbers import Integral, Real

import numpy as np
from scipy.fft import dctn, idctn

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Spread_Spectrum_Embeddings

_MAX_PAYLOAD_BITS = 64


def _validate_image(image) -> np.ndarray:
    """Validates and returns the shared RGB image contract."""
    image_array = np.asarray(image)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f"input_image must have shape (H, W, 3), got {image_array.shape}")
    if image_array.dtype != np.uint8:
        raise TypeError(f"input_image must have dtype uint8, got {image_array.dtype}")
    return image_array


def _validate_watermark(watermark) -> np.ndarray:
    """Validates the binary payload used by embedding solutions."""
    watermark_array = np.asarray(watermark)
    if watermark_array.ndim != 1:
        raise ValueError(f"watermark_bits must be one-dimensional, got shape {watermark_array.shape}")
    if watermark_array.dtype != np.uint8:
        raise TypeError(f"watermark_bits must have dtype uint8, got {watermark_array.dtype}")
    if watermark_array.size == 0:
        raise ValueError("watermark_bits must not be empty")
    if np.any((watermark_array != 0) & (watermark_array != 1)):
        raise ValueError("watermark_bits must contain only 0 and 1")
    return watermark_array


def _validate_integer(name: str, value, minimum: int) -> int:
    """Validates an integer algorithm parameter."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    validated = int(value)
    if validated < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return validated


def _validate_attack_angle(value) -> float:
    """Validates the ABW attack angle in degrees."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("attack_angle must be a real number")
    angle = float(value)
    if not np.isfinite(angle):
        raise ValueError("attack_angle must be finite")
    if not 0.0 < angle < 90.0:
        raise ValueError("attack_angle must be between 0 and 90 degrees")
    return angle


def _to_luma(image: np.ndarray) -> np.ndarray:
    """Converts RGB uint8 to BT.601 luma without rounding."""
    image_float = image.astype(np.float64)
    return 0.299 * image_float[..., 0] + 0.587 * image_float[..., 1] + 0.114 * image_float[..., 2]


def _block_geometry(image_shape: tuple, num_blocks: int) -> tuple:
    """Builds an exact square block grid over an arbitrary rectangular image."""
    height, width = image_shape[:2]
    if num_blocks > min(height, width):
        raise ValueError(
            f"num_blocks is too large: need at least {num_blocks}x{num_blocks} pixels, got {height}x{width}"
        )

    row_edges = np.arange(num_blocks + 1, dtype=np.int64) * height // num_blocks
    column_edges = np.arange(num_blocks + 1, dtype=np.int64) * width // num_blocks
    block_heights = np.diff(row_edges)
    block_widths = np.diff(column_edges)
    areas = block_heights[:, None] * block_widths[None, :]
    return row_edges, column_edges, areas


def _dc_matrix(channel: np.ndarray, num_blocks: int) -> tuple:
    """Computes every block's orthonormal DCT DC coefficient via integral sums."""
    row_edges, column_edges, areas = _block_geometry(channel.shape, num_blocks)
    integral = np.pad(channel.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    block_sums = (
        integral[row_edges[1:, None], column_edges[None, 1:]]
        - integral[row_edges[:-1, None], column_edges[None, 1:]]
        - integral[row_edges[1:, None], column_edges[None, :-1]]
        + integral[row_edges[:-1, None], column_edges[None, :-1]]
    )
    return block_sums / np.sqrt(areas), row_edges, column_edges, areas


def _expand_dc_delta(
    dc_delta: np.ndarray,
    row_edges: np.ndarray,
    column_edges: np.ndarray,
    areas: np.ndarray,
) -> np.ndarray:
    """Converts DC-coefficient changes into their uniform per-block pixel changes."""
    block_delta = dc_delta / np.sqrt(areas)
    row_labels = np.repeat(np.arange(dc_delta.shape[0]), np.diff(row_edges))
    column_labels = np.repeat(np.arange(dc_delta.shape[1]), np.diff(column_edges))
    return block_delta[row_labels[:, None], column_labels[None, :]]


@lru_cache(maxsize=32)
def _zigzag_indices(rows: int, columns: int) -> np.ndarray:
    """Returns flat indices from the lowest to the highest 2-D DCT frequency."""
    coordinates = []
    for diagonal in range(rows + columns - 1):
        first_row = max(0, diagonal - columns + 1)
        last_row = min(rows - 1, diagonal)
        current = [(row, diagonal - row) for row in range(first_row, last_row + 1)]
        if diagonal % 2 == 0:
            current.reverse()
        coordinates.extend(current)
    indices = np.fromiter((row * columns + column for row, column in coordinates), dtype=np.intp)
    indices.flags.writeable = False
    return indices


@lru_cache(maxsize=32)
def _code_matrix(spreading_length: int, key: int, num_codes: int) -> np.ndarray:
    """Builds a reproducible orthogonal Gaussian code matrix."""
    generator = np.random.Generator(np.random.PCG64(key))
    # Generate one full sequence at a time, so decoding a payload prefix uses
    # exactly the same leading sequences as decoding the complete payload.
    gaussian = generator.standard_normal((num_codes, spreading_length)).T
    orthogonal, triangular = np.linalg.qr(gaussian)
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    orthogonal *= signs
    codes = np.ascontiguousarray(np.sqrt(spreading_length) * orthogonal.T)
    codes.flags.writeable = False
    return codes


def _dc_spectrum(image: np.ndarray, num_blocks: int) -> np.ndarray:
    """Performs the second-level DCT over the matrix of block DC terms."""
    dc_matrix, _, _, _ = _dc_matrix(_to_luma(image), num_blocks)
    return dctn(dc_matrix, norm="ortho")


def _location(dc_spectrum: np.ndarray, spreading_length: int) -> tuple:
    """Selects the high-frequency, low-energy end of the ADC zigzag sequence."""
    if spreading_length > dc_spectrum.size:
        raise ValueError(
            f"spreading_length is too large: need {spreading_length} coefficients, available {dc_spectrum.size}"
        )
    indices = _zigzag_indices(*dc_spectrum.shape)[-spreading_length:]
    return dc_spectrum.reshape(-1)[indices], indices


def _decode_scores(
    image: np.ndarray,
    num_bits: int,
    spreading_length: int,
    num_blocks: int,
    key: int,
) -> np.ndarray:
    """Returns spreading-code correlations for a validated image and profile."""
    location, _ = _location(_dc_spectrum(image, num_blocks), spreading_length)
    codes = _code_matrix(spreading_length, key, num_bits)
    return codes @ location / spreading_length


def _decode_bits(
    image: np.ndarray,
    num_bits: int,
    spreading_length: int,
    num_blocks: int,
    key: int,
) -> np.ndarray:
    """Decodes bits from a validated image and profile."""
    scores = _decode_scores(image, num_bits, spreading_length, num_blocks, key)
    return np.ascontiguousarray((scores >= 0.0).astype(np.int8))


class ADC_ABW(Ready_Spread_Spectrum_Embeddings):
    """Many-to-one Adjusted-DC spread spectrum with attack-based weight.

    The defaults use the published ADC-ABW4 parameters: a 128x128 block
    grid, a 320-coefficient location, a 30-degree attack angle, and the 64-bit
    payload size used in the evaluation. As in the paper, this profile has no
    rotation or crop synchronization; scaling performance also depends on the
    image resolution and resampling alignment.
    """

    @staticmethod
    def embedding(**args):
        """Embeds a binary payload using the 2025 ADC-ABW construction.

        Args:
            input_image (np.ndarray): RGB uint8 image of shape (H, W, 3).
            watermark_bits (np.ndarray): Non-empty one-dimensional uint8 array of up to 64 bits.
            attack_angle (float): ABW angle in degrees. Smaller values improve robustness at the
                expense of visual quality; the paper determines it from the expected dominant attack.
            spreading_length (int): Number of high-frequency coefficients in the ADC location.
            num_blocks (int): Number of blocks along each image axis. Both image dimensions must be
                at least this large. This value is part of the extraction key.
            key (int): Non-negative seed for the orthogonal spreading codes. It is not a cryptographic key.

        Returns:
            np.ndarray: C-contiguous RGB uint8 watermarked image.
        """
        defaults = {
            "input_image": None,
            "watermark_bits": None,
            "attack_angle": 30.0,
            "spreading_length": 320,
            "num_blocks": 128,
            "key": 0,
        }
        args = {**defaults, **args}

        if args["input_image"] is None or args["watermark_bits"] is None:
            raise ValueError("input_image and watermark_bits are required")

        image = _validate_image(args["input_image"])
        watermark = _validate_watermark(args["watermark_bits"])
        attack_angle = _validate_attack_angle(args["attack_angle"])
        spreading_length = _validate_integer("spreading_length", args["spreading_length"], 2)
        num_blocks = _validate_integer("num_blocks", args["num_blocks"], 1)
        key = _validate_integer("key", args["key"], 0)

        payload_capacity = min(spreading_length, _MAX_PAYLOAD_BITS)
        if watermark.size > payload_capacity:
            raise ValueError(f"Not enough payload capacity: need {watermark.size} bits, available {payload_capacity}")

        input_luma = _to_luma(image)
        dc_matrix, row_edges, column_edges, areas = _dc_matrix(input_luma, num_blocks)
        dc_spectrum = dctn(dc_matrix, norm="ortho")
        location, indices = _location(dc_spectrum, spreading_length)
        codes = _code_matrix(spreading_length, key, watermark.size)
        bipolar_watermark = 2.0 * watermark.astype(np.float64) - 1.0
        spread_watermark = bipolar_watermark @ codes

        location_norm = np.linalg.norm(location)
        spectrum_norm = np.linalg.norm(dc_spectrum)
        if location_norm <= 1e-12 * max(1.0, spectrum_norm):
            raise ValueError("selected ADC location has no energy; the image cannot carry this watermark")
        watermark_norm = np.linalg.norm(spread_watermark)
        weight = location_norm / (np.tan(np.deg2rad(attack_angle)) * watermark_norm)
        modified_spectrum = dc_spectrum.copy().reshape(-1)
        modified_spectrum[indices] = location + weight * spread_watermark
        modified_dc = idctn(modified_spectrum.reshape(dc_spectrum.shape), norm="ortho")

        luma_delta = _expand_dc_delta(modified_dc - dc_matrix, row_edges, column_edges, areas)
        output = image.astype(np.float64) + luma_delta[..., None]
        output = np.ascontiguousarray(np.clip(np.rint(output), 0, 255).astype(np.uint8))

        if np.array_equal(output, image):
            raise ValueError(
                "watermark is erased by uint8 quantization; decrease attack_angle or adjust the ADC profile"
            )

        original_scores = codes @ location / spreading_length
        output_scores = _decode_scores(output, watermark.size, spreading_length, num_blocks, key)
        realized_gain = np.mean(bipolar_watermark * (output_scores - original_scores))
        if realized_gain < 0.1 * weight:
            raise ValueError(
                "watermark is too weak after uint8 quantization; decrease attack_angle or adjust the ADC profile"
            )

        decoded = np.ascontiguousarray((output_scores >= 0.0).astype(np.int8))
        if not np.array_equal(decoded.astype(np.uint8), watermark):
            raise ValueError(
                "watermark does not survive uint8 quantization; decrease attack_angle or adjust the ADC profile"
            )
        return output

    @staticmethod
    def extraction(**args):
        """Blindly extracts a payload using the same ADC location and code key.

        Args:
            input_image (np.ndarray): RGB uint8 image containing a watermark.
            num_bits (int): Number of payload bits to extract, at most 64.
            spreading_length (int): Value used during embedding.
            num_blocks (int): Value used during embedding.
            key (int): Seed used during embedding.

        Returns:
            np.ndarray: C-contiguous int8 array containing 0/1 values.
        """
        defaults = {
            "input_image": None,
            "num_bits": 0,
            "spreading_length": 320,
            "num_blocks": 128,
            "key": 0,
        }
        args = {**defaults, **args}

        if args["input_image"] is None:
            raise ValueError("input_image is required")

        image = _validate_image(args["input_image"])
        num_bits = _validate_integer("num_bits", args["num_bits"], 1)
        spreading_length = _validate_integer("spreading_length", args["spreading_length"], 2)
        num_blocks = _validate_integer("num_blocks", args["num_blocks"], 1)
        key = _validate_integer("key", args["key"], 0)

        payload_capacity = min(spreading_length, _MAX_PAYLOAD_BITS)
        if num_bits > payload_capacity:
            raise ValueError(f"Cannot extract {num_bits} bits: payload capacity is {payload_capacity}")

        return _decode_bits(image, num_bits, spreading_length, num_blocks, key)
