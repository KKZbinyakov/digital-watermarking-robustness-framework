"""
Метод Contourlet Transform — многонаправленное разложение
https://www.scirp.org/html/747.html
(с небольшими изменениями)
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

from dwarf.core.embedding_orchestrator.embedding_core import Ready_Frequency_Embeddings
from dwarf.ready_solutions.utils.embedding_utils_pyx cimport (
    init_filters, init_offsets, contourlet_decompose, contourlet_reconstruct,
    capacity, embed_subband, extract_subband, count_subband_bit_errors,
    valid_contourlet_shape
)

cnp.import_array()

def _embed_core(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] image,
                cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] watermark,
                double margin, int n_levels, int dfb_levels,
                int sc, int block, int iterations):
    """
    Ядро встраивания ЦВЗ.
    """
    cdef int wm_len = watermark.shape[0]
    cdef cnp.int32_t[:] wm_view = watermark
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] cur = image
    cdef double[:, :] S
    cdef int s_i, n_sub, h, w, cap, it, errors

    for it in range(iterations):
        lowpass, bands = contourlet_decompose(cur, n_levels, dfb_levels)
        subbands = bands[sc]
        n_sub = len(subbands)
        h = subbands[0].shape[0]
        w = subbands[0].shape[1]

        if it == 0:
            cap = capacity(h, w, block, n_sub)
            if wm_len > cap:
                raise ValueError(
                    f"Given {wm_len} bits, but capacity of the scale {sc} is only {cap} bits"
                    f"({n_sub} subbands {h}x{w}, block {block}x{block}).")
        else:
            # Проверяем предыдущий проход уже после реконструкции. Если все
            # биты читаются верно, дальнейшее усиление не требуется.
            errors = count_subband_bit_errors(subbands, wm_view, wm_len, block)
            if errors == 0:
                return cur

        for s_i in range(n_sub):
            S = subbands[s_i]
            with nogil:
                embed_subband(S, wm_view, wm_len, s_i, n_sub, margin, block)

        cur = np.ascontiguousarray(
            contourlet_reconstruct(np.ascontiguousarray(lowpass, dtype=np.float64),
                                   bands, dfb_levels),
            dtype=np.float64)

    # Проверяем состояние после последней разрешённой итерации.
    lowpass, bands = contourlet_decompose(cur, n_levels, dfb_levels)
    subbands = bands[sc]
    errors = count_subband_bit_errors(subbands, wm_view, wm_len, block)
    if errors != 0:
        raise RuntimeError(
            f"Contourlet embedding did not converge after {iterations} iterations: "
            f"{errors} of {wm_len} bits are not recoverable."
        )

    return cur

def _extract_core(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] image,
                  int wm_length, int n_levels, int dfb_levels,
                  int sc, int block):
    """
    Ядро извлечения ЦВЗ.
    """
    lowpass, bands = contourlet_decompose(image, n_levels, dfb_levels)
    subbands = bands[sc]

    cdef int n_sub = len(subbands)
    cdef int h = subbands[0].shape[0]
    cdef int w = subbands[0].shape[1]
    cdef int cap = capacity(h, w, block, n_sub)

    if wm_length > cap:
        raise ValueError(
            f"Given {wm_length} bits, but capacity of the scale {sc} is only {cap} bits.")

    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] extracted_wm = np.zeros(
        wm_length, dtype=np.int32)
    cdef cnp.int32_t[:] wm_view = extracted_wm

    cdef double[:, :] S
    cdef int s_i
    for s_i in range(n_sub):
        S = subbands[s_i]
        with nogil:
            extract_subband(S, wm_view, wm_length, s_i, n_sub, block)

    return extracted_wm


class Contourlet(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        """
        Встраивает биты ЦВЗ в контурлет-коэффициенты изображения.
        :param input_image: RGB-изображение uint8 формы (H, W, 3).
        :param watermark_bits: массив uint8 из значений 0/1.
        :param margin: требуемый зазор между парой коэффициентов.
        :param n_levels: число масштабов Лапласовой пирамиды.
        :param dfb_levels: число уровней направленного дерева. Если исходное
            изображение не совместимо с ограничениями DFB, используется максимальная
            квадратная верхняя левая область со стороной, кратной
            2**(n_levels + dfb_levels); остальная часть изображения сохраняется
            без изменений.
        :param scale: индекс масштаба для встраивания (-1 для последнего).
        :param block: сторона блока внутри поддиапазона.
        :param iterations: максимальное число итераций встраивания. Процесс
            завершается раньше, как только все биты корректно читаются после
            reconstruction -> decomposition.

        :return output_image: RGB-изображение uint8 формы (H, W, 3) со встроенным ЦВЗ.
        """
        defaults = {
                    "input_image": None,
                    "watermark_bits": None,
                    "margin": 24.0,
                    "n_levels": 2,
                    "dfb_levels": 2,
                    "scale": -1,
                    "block": 4,
                    "iterations": 10
                }
        args = {**defaults, **args}
        image = args["input_image"]
        watermark = args["watermark_bits"]
        if image is None or watermark is None:
            raise ValueError("input_image/image_path or watermark_bits not given")

        image_arr = np.asarray(image)
        if image_arr.ndim != 3 or image_arr.shape[2] != 3:
            raise ValueError(
                f"input_image must have shape (H, W, 3), got {image_arr.shape}"
            )
        if image_arr.dtype != np.uint8:
            raise TypeError(
                f"input_image must have dtype uint8, got {image_arr.dtype}"
            )

        watermark_arr = np.asarray(watermark)
        if watermark_arr.ndim != 1:
            raise ValueError(
                f"watermark_bits must be one-dimensional, got shape {watermark_arr.shape}"
            )
        if watermark_arr.dtype != np.uint8:
            raise TypeError(
                f"watermark_bits must have dtype uint8, got {watermark_arr.dtype}"
            )
        if watermark_arr.size == 0:
            raise ValueError("watermark_bits must not be empty")
        if np.any((watermark_arr != 0) & (watermark_arr != 1)):
            raise ValueError("watermark_bits must contain only 0 and 1")

        cdef cnp.ndarray[cnp.uint8_t, ndim=3, mode='c'] rgb_c = np.ascontiguousarray(image_arr)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] input_y = np.ascontiguousarray(
            0.299 * rgb_c[:, :, 0]
            + 0.587 * rgb_c[:, :, 1]
            + 0.114 * rgb_c[:, :, 2],
            dtype=np.float64,
        )
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = input_y.copy()
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] wm_c = np.ascontiguousarray(
            watermark_arr, dtype=np.int32
        )

        cdef double margin = args["margin"]
        cdef int n_levels = int(args["n_levels"])
        cdef int dfb_levels = int(args["dfb_levels"])
        cdef int scale = int(args["scale"])
        cdef int block = int(args["block"])
        cdef int iterations = int(args["iterations"])
        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]
        cdef int valid_h, valid_w

        if n_levels < 1:
            raise ValueError("n_levels must be >= 1.")
        if dfb_levels < 0:
            raise ValueError("dfb_levels must be >= 0.")

        cdef int sc = n_levels - 1 if scale < 0 else scale
        if sc < 0 or sc >= n_levels:
            raise ValueError(f"scale={sc} out of range [0, {n_levels - 1}].")
        if block < 4:
            raise ValueError("block must be >= 4.")
        if iterations < 1:
            raise ValueError("iterations must be >= 1.")

        valid_h, valid_w = valid_contourlet_shape(H, W, n_levels, dfb_levels)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] work = np.ascontiguousarray(
            img_c[:valid_h, :valid_w], dtype=np.float64
        )

        init_filters()
        init_offsets()

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] embedded = _embed_core(
            work, wm_c, margin, n_levels, dfb_levels, sc, block, iterations
        )

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] watermarked_y
        if valid_h == H and valid_w == W:
            watermarked_y = embedded
        else:
            watermarked_y = img_c.copy()
            watermarked_y[:valid_h, :valid_w] = embedded

        delta_y = watermarked_y - input_y
        output_rgb = rgb_c.astype(np.float64) + delta_y[:, :, None]
        return np.ascontiguousarray(
            np.clip(np.rint(output_rgb), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def extraction(**args):
        """
        Извлекает биты ЦВЗ из контурлет-коэффициентов изображения.
        :param input_image: RGB-изображение uint8 формы (H, W, 3) с ЦВЗ.
        :param num_bits: длина ЦВЗ в битах.
        :param n_levels: число масштабов лапласовой пирамиды.
        :param dfb_levels: число уровней направленного дерева. Используется та же
            максимальная квадратная верхняя левая область со стороной, кратной
            2**(n_levels + dfb_levels), что и при embedding.
        :param scale: индекс масштаба.
        :param block: сторона блока внутри поддиапазона.

        :return extracted_wm: извлечённые биты ЦВЗ.
        """
        defaults = {
                    "input_image": None,
                    "num_bits": 0,
                    "n_levels": 2,
                    "dfb_levels": 2,
                    "scale": -1,
                    "block": 4
                }
        args = {**defaults, **args}
        image = args["input_image"]
        num_bits = args["num_bits"]
        if image is None or not num_bits:
            raise ValueError("input_image/image_path or num_bits not given")

        image_arr = np.asarray(image)
        if image_arr.ndim != 3 or image_arr.shape[2] != 3:
            raise ValueError(
                f"input_image must have shape (H, W, 3), got {image_arr.shape}"
            )
        if image_arr.dtype != np.uint8:
            raise TypeError(
                f"input_image must have dtype uint8, got {image_arr.dtype}"
            )

        cdef cnp.ndarray[cnp.uint8_t, ndim=3, mode='c'] rgb_c = np.ascontiguousarray(image_arr)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(
            0.299 * rgb_c[:, :, 0]
            + 0.587 * rgb_c[:, :, 1]
            + 0.114 * rgb_c[:, :, 2],
            dtype=np.float64,
        )
        cdef int wm_length = num_bits

        cdef int n_levels = int(args["n_levels"])
        cdef int dfb_levels = int(args["dfb_levels"])
        cdef int scale = int(args["scale"])
        cdef int block = int(args["block"])
        cdef int H = img_c.shape[0]
        cdef int W = img_c.shape[1]
        cdef int valid_h, valid_w

        if n_levels < 1:
            raise ValueError("n_levels must be >= 1.")
        if dfb_levels < 0:
            raise ValueError("dfb_levels must be >= 0.")

        cdef int sc = n_levels - 1 if scale < 0 else scale
        if sc < 0 or sc >= n_levels:
            raise ValueError(f"scale={sc} out of range [0, {n_levels - 1}].")
        if block < 4:
            raise ValueError("block must be >= 4.")

        valid_h, valid_w = valid_contourlet_shape(H, W, n_levels, dfb_levels)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] work = np.ascontiguousarray(
            img_c[:valid_h, :valid_w], dtype=np.float64
        )

        init_filters()
        init_offsets()

        extracted_wm = _extract_core(work, wm_length, n_levels, dfb_levels, sc, block)
        if np.any((extracted_wm != 0) & (extracted_wm != 1)):
            raise RuntimeError("contourlet extraction produced a non-binary watermark")
        return np.ascontiguousarray(extracted_wm, dtype=np.int8)
