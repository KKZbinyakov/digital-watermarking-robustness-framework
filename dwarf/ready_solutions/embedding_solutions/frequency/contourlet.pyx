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
    init_filters, init_offsets, contourlet_decompose, contourlet_reconstruct
)

cnp.import_array()

cdef inline int capacity(int h, int w, int block, int n_sub) noexcept nogil:
    """
    Ёмкость масштаба: число блоков во всех его направленных поддиапазонах.

    Args:
        h: высота направленного поддиапазона
        w: ширина направленного поддиапазона
        block: сторона блока
        n_sub: число направленных поддиапазонов масштаба

    Returns:
        максимальное число бит, которое можно встроить в данный масштаб
    """
    return n_sub * (h // block) * (w // block)

cdef void embed_subband(double[:, :] S, cnp.int32_t[:] wm, int wm_len,
                         int sub_i, int n_sub, double margin,
                         int block) noexcept nogil:
    """
    Встраивание бит в один направленный поддиапазон.

    Args:
        S: направленный поддиапазон.
        wm: биты ЦВЗ.
        wm_len: длина ЦВЗ.
        sub_i: номер данного поддиапазона.
        n_sub: общее число поддиапазонов масштаба.
        margin: требуемый зазор между парой коэффициентов.
        block: сторона блока
    """
    cdef int h = S.shape[0]
    cdef int w = S.shape[1]
    cdef int nb_c = w // block
    cdef int n_blocks = (h // block) * nb_c
    cdef int blk, i_bit, br, bc, r0, k0
    cdef double c1, c2, d

    for blk in range(n_blocks):
        i_bit = blk * n_sub + sub_i
        if i_bit >= wm_len:
            return

        br = blk // nb_c
        bc = blk % nb_c
        r0 = br * block
        k0 = bc * block

        c1 = S[r0 + 1, k0 + 1]
        c2 = S[r0 + 2, k0 + 2]

        if wm[i_bit] == 1:
            if c1 - c2 < margin:
                d = 0.5 * (margin - (c1 - c2))
                S[r0 + 1, k0 + 1] = c1 + d
                S[r0 + 2, k0 + 2] = c2 - d
        else:
            if c2 - c1 < margin:
                d = 0.5 * (margin - (c2 - c1))
                S[r0 + 2, k0 + 2] = c2 + d
                S[r0 + 1, k0 + 1] = c1 - d

cdef void extract_subband(double[:, :] S, cnp.int32_t[:] wm, int wm_len,
                           int sub_i, int n_sub, int block) noexcept nogil:
    """
    Извлечение бит из одного направленного поддиапазона.

    Args:
        S: направленный поддиапазон изображения с ЦВЗ.
        wm: выходной массив бит ЦВЗ.
        wm_len: длина ЦВЗ.
        sub_i: номер данного поддиапазона.
        n_sub: общее число поддиапазонов масштаба.
        block: сторона блока;
            обязана совпадать со значением, использованным в embed_subband.
    """
    cdef int h = S.shape[0]
    cdef int w = S.shape[1]
    cdef int nb_c = w // block
    cdef int n_blocks = (h // block) * nb_c
    cdef int blk, i_bit, br, bc, r0, k0
    cdef double c1, c2

    for blk in range(n_blocks):
        i_bit = blk * n_sub + sub_i
        if i_bit >= wm_len:
            return

        br = blk // nb_c
        bc = blk % nb_c
        r0 = br * block
        k0 = bc * block

        c1 = S[r0 + 1, k0 + 1]
        c2 = S[r0 + 2, k0 + 2]

        wm[i_bit] = 1 if c1 > c2 else 0

cdef int count_subband_bit_errors(object subbands, cnp.int32_t[:] wm,
                                  int wm_len, int block):
    """Считает число бит, потерянных после reconstruction -> decomposition."""
    cdef int n_sub = len(subbands)
    cdef int s_i, h, w, nb_c, n_blocks
    cdef int blk, i_bit, br, bc, r0, k0
    cdef int errors = 0
    cdef double[:, :] S
    cdef double c1, c2

    for s_i in range(n_sub):
        S = subbands[s_i]
        h = S.shape[0]
        w = S.shape[1]
        nb_c = w // block
        n_blocks = (h // block) * nb_c

        for blk in range(n_blocks):
            i_bit = blk * n_sub + s_i
            if i_bit >= wm_len:
                break

            br = blk // nb_c
            bc = blk % nb_c
            r0 = br * block
            k0 = bc * block

            c1 = S[r0 + 1, k0 + 1]
            c2 = S[r0 + 2, k0 + 2]

            if (wm[i_bit] == 1 and c1 <= c2) or (wm[i_bit] != 1 and c2 <= c1):
                errors += 1

    return errors

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


def _valid_contourlet_shape(int height, int width, int n_levels, int dfb_levels):
    """
    Возвращает максимальную квадратную верхнюю левую область, совместимую
    с текущей реализацией Лапласовой пирамиды и DFB.

    DFB в embedding_utils_pyx предъявляет не только требование кратности
    размеров степени двойки, но и ограничение на их взаимное отношение.
    Квадратная область гарантированно удовлетворяет этому ограничению на
    каждом уровне разложения. Сторона дополнительно делается кратной
    2**(n_levels + dfb_levels).
    """
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1.")
    if dfb_levels < 0:
        raise ValueError("dfb_levels must be >= 0.")

    cdef int factor = 1 << (n_levels + dfb_levels)
    cdef int side = (min(height, width) // factor) * factor

    if side == 0:
        raise ValueError(
            f"Image size {width}x{height} is too small for "
            f"n_levels={n_levels}, dfb_levels={dfb_levels}: "
            f"need at least {factor}x{factor}."
        )

    return side, side


class Contourlet(Ready_Frequency_Embeddings):
    @staticmethod
    def embedding(**args):
        """
        Встраивает биты ЦВЗ в контурлет-коэффициенты изображения.
        :param input_image: матрица входного изображения (канал яркости Y).
        :param watermark_bits: массив битов ЦВЗ.
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

        :return output_image: матрица изображения с встроенным ЦВЗ.
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

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(image, dtype=np.float64)
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] wm_c = np.ascontiguousarray(watermark, dtype=np.int32)

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

        valid_h, valid_w = _valid_contourlet_shape(H, W, n_levels, dfb_levels)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] work = np.ascontiguousarray(
            img_c[:valid_h, :valid_w], dtype=np.float64
        )

        init_filters()
        init_offsets()

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] embedded = _embed_core(
            work, wm_c, margin, n_levels, dfb_levels, sc, block, iterations
        )

        if valid_h == H and valid_w == W:
            return embedded

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] output = img_c.copy()
        output[:valid_h, :valid_w] = embedded
        return output

    @staticmethod
    def extraction(**args):
        """
        Извлекает биты ЦВЗ из контурлет-коэффициентов изображения.
        :param input_image: матрица изображения с ЦВЗ (канал яркости Y).
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
        if image is None or num_bits is None:
            raise ValueError("input_image/image_path or num_bits not given")

        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] img_c = np.ascontiguousarray(image, dtype=np.float64)
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

        valid_h, valid_w = _valid_contourlet_shape(H, W, n_levels, dfb_levels)
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] work = np.ascontiguousarray(
            img_c[:valid_h, :valid_w], dtype=np.float64
        )

        init_filters()
        init_offsets()

        return _extract_core(work, wm_length, n_levels, dfb_levels, sc, block)
