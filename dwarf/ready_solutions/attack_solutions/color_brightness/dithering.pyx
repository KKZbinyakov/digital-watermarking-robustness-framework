"""Атака дизеринга: диффузия ошибки по Флойду-Стейнбергу или упорядоченный
дизеринг по матрице Байера.
"""

import numpy as np

cimport cython
cimport numpy as cnp
from libc.math cimport floor

from dwarf.core.attack_orchestrator.attack_core import Ready_Color_Brightness_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_matrix

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _floyd_steinberg(double[:, :, ::1] buffer, int levels) noexcept nogil:
    """
    Квантует изображение с диффузией ошибки по Флойду-Стейнбергу.

    Ошибка квантования текущего пикселя распределяется по ещё не обработанным
    соседям с весами 7/16 вправо, 3/16 влево-вниз, 5/16 вниз и 1/16 вправо-вниз.
    Алгоритм последовательный: значение пикселя зависит от уже обработанных
    соседей, поэтому векторизовать его средствами numpy нельзя.

    Args:
        buffer (double[:, :, ::1]): изображение (H, W, 3) со значениями 0..255, меняется на месте
        levels (int): число уровней квантования на канал, не меньше 2

    Returns:
        None
    """
    cdef Py_ssize_t height = buffer.shape[0]
    cdef Py_ssize_t width = buffer.shape[1]
    cdef Py_ssize_t y, x, channel
    cdef double step = 255.0 / (levels - 1)
    cdef double old_value, new_value, error

    for y in range(height):
        for x in range(width):
            for channel in range(3):
                old_value = buffer[y, x, channel]
                new_value = floor(old_value / step + 0.5) * step
                if new_value < 0.0:
                    new_value = 0.0
                elif new_value > 255.0:
                    new_value = 255.0
                buffer[y, x, channel] = new_value
                error = old_value - new_value

                if x + 1 < width:
                    buffer[y, x + 1, channel] += error * 0.4375
                if y + 1 < height:
                    if x > 0:
                        buffer[y + 1, x - 1, channel] += error * 0.1875
                    buffer[y + 1, x, channel] += error * 0.3125
                    if x + 1 < width:
                        buffer[y + 1, x + 1, channel] += error * 0.0625


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _ordered(double[:, :, ::1] buffer, double[:, ::1] matrix, int levels) noexcept nogil:
    """
    Квантует изображение с порогом из тайлящейся матрицы Байера.

    В отличие от диффузии ошибки атака детерминирована и вносит регулярный
    периодический паттерн, что по-разному влияет на частотные схемы встраивания.

    Args:
        buffer (double[:, :, ::1]): изображение (H, W, 3) со значениями 0..255, меняется на месте
        matrix (double[:, ::1]): нормированная матрица Байера (n, n) со значениями в [0, 1)
        levels (int): число уровней квантования на канал, не меньше 2

    Returns:
        None
    """
    cdef Py_ssize_t height = buffer.shape[0]
    cdef Py_ssize_t width = buffer.shape[1]
    cdef Py_ssize_t size = matrix.shape[0]
    cdef Py_ssize_t y, x, channel
    cdef double step = 255.0 / (levels - 1)
    cdef double threshold, quantized

    for y in range(height):
        for x in range(width):
            threshold = matrix[y % size, x % size] - 0.5
            for channel in range(3):
                quantized = floor(buffer[y, x, channel] / step + threshold + 0.5) * step
                if quantized < 0.0:
                    quantized = 0.0
                elif quantized > 255.0:
                    quantized = 255.0
                buffer[y, x, channel] = quantized


def _bayer_matrix(int size):
    """
    Строит нормированную матрицу Байера рекурсивным удвоением.

    Args:
        size (int): сторона матрицы, степень двойки не меньше 2

    Returns:
        np.ndarray: непрерывный массив (size, size) типа float64 со значениями в [0, 1)
    """
    matrix = np.array([[0, 2], [3, 1]], dtype=np.float64)
    while matrix.shape[0] < size:
        matrix = np.block([[4 * matrix, 4 * matrix + 2],
                           [4 * matrix + 3, 4 * matrix + 1]])
    return np.ascontiguousarray(matrix / matrix.size)


class Dithering(Ready_Color_Brightness_Attacks):
    """
    Атака дизеринга.

    Огрубляет изображение до небольшого числа уровней на канал, компенсируя
    потерю градаций пространственным шумом. Разрушает ЦВЗ, закодированные в
    младших битах и в локальных перепадах яркости.
    """

    @staticmethod
    def attack(**args):
        """
        Применяет дизеринг к изображению.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                method (str): 'floyd_steinberg' или 'ordered' (по умолчанию 'floyd_steinberg')
                levels (int): число уровней на канал, не меньше 2 (по умолчанию 2)
                matrix_size (int): сторона матрицы Байера для 'ordered', степень двойки (по умолчанию 4)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если levels меньше 2, matrix_size меньше 2 или method неизвестен
        """
        defaults = {
            "input_image": None,
            "method": "floyd_steinberg",
            "levels": 2,
            "matrix_size": 4,
        }
        args = {**defaults, **args}
        input_image = args["input_image"]
        method = args["method"]
        cdef int levels = int(args["levels"])
        cdef int matrix_size = int(args["matrix_size"])
        cdef double[:, :, ::1] view
        cdef double[:, ::1] matrix_view

        if levels < 2:
            raise ValueError(f"levels must be at least 2, got {levels}")

        buffer = np.ascontiguousarray(to_matrix(input_image), dtype=np.float64)
        view = buffer

        if method == "floyd_steinberg":
            with nogil:
                _floyd_steinberg(view, levels)
        elif method == "ordered":
            if matrix_size < 2:
                raise ValueError(
                    f"matrix_size must be at least 2, got {matrix_size}"
                )
            matrix_view = _bayer_matrix(matrix_size)
            with nogil:
                _ordered(view, matrix_view, levels)
        else:
            raise ValueError(
                f"unknown method={method!r}, expected 'floyd_steinberg' or 'ordered'"
            )

        return to_matrix(buffer)
