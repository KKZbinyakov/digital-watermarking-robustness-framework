"""Атака уменьшения числа цветов: медианное сечение Pillow или k-means."""

import numpy as np

cimport cython
cimport numpy as cnp

from PIL import Image

from dwarf.core.attack_orchestrator.attack_core import Ready_Color_Brightness_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_array, to_matrix, to_pil

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _assign(double[:, ::1] pixels,
                  double[:, ::1] centers,
                  Py_ssize_t[::1] labels) noexcept nogil:
    """
    Приписывает каждому пикселю индекс ближайшего центра палитры.

    Расстояние считается в цикле, без построения матрицы разностей размера
    (число пикселей) x (число центров) x 3, поэтому потребление памяти не
    зависит от размера палитры.

    Args:
        pixels (double[:, ::1]): пиксели (N, 3) со значениями 0..255
        centers (double[:, ::1]): центры палитры (K, 3)
        labels (Py_ssize_t[::1]): выходной массив длины N, заполняется индексами центров

    Returns:
        None
    """
    cdef Py_ssize_t count = pixels.shape[0]
    cdef Py_ssize_t palette = centers.shape[0]
    cdef Py_ssize_t index, center, best
    cdef double distance, best_distance, difference

    for index in range(count):
        best = 0
        best_distance = 0.0
        for center in range(palette):
            distance = 0.0
            difference = pixels[index, 0] - centers[center, 0]
            distance += difference * difference
            difference = pixels[index, 1] - centers[center, 1]
            distance += difference * difference
            difference = pixels[index, 2] - centers[center, 2]
            distance += difference * difference
            if center == 0 or distance < best_distance:
                best_distance = distance
                best = center
        labels[index] = best


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _accumulate(double[:, ::1] pixels,
                      Py_ssize_t[::1] labels,
                      double[:, ::1] sums,
                      Py_ssize_t[::1] counts) noexcept nogil:
    """
    Складывает пиксели по кластерам для пересчёта центров.

    Args:
        pixels (double[:, ::1]): пиксели (N, 3)
        labels (Py_ssize_t[::1]): индексы кластеров длины N
        sums (double[:, ::1]): выходные суммы (K, 3), должны быть обнулены до вызова
        counts (Py_ssize_t[::1]): выходные размеры кластеров (K,), должны быть обнулены до вызова

    Returns:
        None
    """
    cdef Py_ssize_t count = pixels.shape[0]
    cdef Py_ssize_t index, label

    for index in range(count):
        label = labels[index]
        sums[label, 0] += pixels[index, 0]
        sums[label, 1] += pixels[index, 1]
        sums[label, 2] += pixels[index, 2]
        counts[label] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
def _fit_kmeans(double[:, ::1] sample, double[:, ::1] centers, int iterations):
    """
    Обучает палитру методом k-means на выборке пикселей.

    Пустые кластеры остаются на прежнем месте: переинициализация случайной
    точкой сделала бы атаку невоспроизводимой при фиксированном seed.

    Args:
        sample (double[:, ::1]): выборка пикселей (M, 3)
        centers (double[:, ::1]): начальные центры (K, 3), меняются на месте
        iterations (int): число итераций

    Returns:
        None
    """
    cdef Py_ssize_t palette = centers.shape[0]
    cdef Py_ssize_t step, center, component

    labels_array = np.empty(sample.shape[0], dtype=np.intp)
    sums_array = np.zeros((palette, 3), dtype=np.float64)
    counts_array = np.zeros(palette, dtype=np.intp)

    cdef Py_ssize_t[::1] labels = labels_array
    cdef double[:, ::1] sums = sums_array
    cdef Py_ssize_t[::1] counts = counts_array

    for step in range(iterations):
        sums_array[:] = 0.0
        counts_array[:] = 0
        with nogil:
            _assign(sample, centers, labels)
            _accumulate(sample, labels, sums, counts)
            for center in range(palette):
                if counts[center] > 0:
                    for component in range(3):
                        centers[center, component] = sums[center, component] / counts[center]


class Color_Quantization(Ready_Color_Brightness_Attacks):
    """
    Атака уменьшения числа цветов.

    Сводит палитру изображения к заданному числу цветов методом медианного
    сечения или k-means. Огрубление палитры стирает малые отклонения яркости,
    которыми кодируется ЦВЗ в пространственных схемах встраивания.
    """

    @staticmethod
    def attack(**args):
        """
        Уменьшает число цветов изображения.

        Палитра k-means обучается на случайной выборке пикселей: на полном кадре
        качество палитры уже не растёт, а время обучения растёт линейно по числу
        пикселей. Финальная разметка при этом идёт по всему изображению.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                colors (int): число цветов в палитре, не меньше 2 (по умолчанию 16)
                method (str): 'median_cut' или 'kmeans' (по умолчанию 'median_cut')
                sample_size (int): размер выборки для обучения k-means (по умолчанию 5000)
                iterations (int): число итераций k-means (по умолчанию 10)
                seed (int): зерно генератора случайных чисел для k-means (по умолчанию None)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            ValueError: если colors меньше 2, colors больше размера выборки или method неизвестен
        """
        defaults = {
            "input_image": None,
            "colors": 16,
            "method": "median_cut",
            "sample_size": 5000,
            "iterations": 10,
            "seed": None,
        }
        args = {**defaults, **args}
        input_image = args["input_image"]
        method = args["method"]
        cdef int colors = int(args["colors"])
        cdef int iterations = int(args["iterations"])
        sample_size = int(args["sample_size"])
        seed = args["seed"]

        if colors < 2:
            raise ValueError(f"colors must be at least 2, got {colors}")

        if method == "median_cut":
            return to_array(to_pil(input_image).quantize(colors=colors, method=Image.MEDIANCUT))

        if method != "kmeans":
            raise ValueError(
                f"unknown method={method!r}, expected 'median_cut' or 'kmeans'"
            )

        rng = np.random.default_rng(seed)
        array = to_matrix(input_image)
        pixels_array = np.ascontiguousarray(array.reshape(-1, 3), dtype=np.float64)

        taken = min(sample_size, pixels_array.shape[0])
        if colors > taken:
            raise ValueError(
                f"colors={colors} exceeds the available sample ({taken} pixels): "
                f"increase sample_size or decrease colors"
            )

        sample_array = np.ascontiguousarray(
            pixels_array[rng.choice(pixels_array.shape[0], taken, replace=False)]
        )
        centers_array = np.ascontiguousarray(
            sample_array[rng.choice(taken, colors, replace=False)]
        )

        _fit_kmeans(sample_array, centers_array, iterations)

        labels_array = np.empty(pixels_array.shape[0], dtype=np.intp)
        cdef double[:, ::1] pixels = pixels_array
        cdef double[:, ::1] centers = centers_array
        cdef Py_ssize_t[::1] labels = labels_array
        with nogil:
            _assign(pixels, centers, labels)

        return to_matrix(centers_array[labels_array].reshape(array.shape))
