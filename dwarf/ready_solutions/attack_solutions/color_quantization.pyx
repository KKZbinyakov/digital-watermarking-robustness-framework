"""Атака уменьшения числа цветов: медианное сечение Pillow или k-means."""

import numpy as np

cimport cython
cimport numpy as cnp

from ..utils.attack_utils import *

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
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Уменьшает число цветов изображения и сохраняет результат.

        Палитра k-means обучается на случайной выборке пикселей: на полном кадре
        качество палитры уже не растёт, а время обучения растёт линейно по числу
        пикселей. Финальная разметка при этом идёт по всему изображению.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                colors (int): число цветов в палитре, не меньше 2 (по умолчанию 16)
                method (str): 'median_cut' или 'kmeans' (по умолчанию 'median_cut')
                sample_size (int): размер выборки для обучения k-means (по умолчанию 5000)
                iterations (int): число итераций k-means (по умолчанию 10)
                seed (int): зерно генератора случайных чисел для k-means (по умолчанию None)

        Returns:
            None

        Raises:
            ValueError: если colors меньше 2, colors больше размера выборки или method неизвестен
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        method = args.get("method", "median_cut")
        cdef int colors = int(args.get("colors", 16))
        cdef int iterations = int(args.get("iterations", 10))
        sample_size = int(args.get("sample_size", 5000))
        seed = args.get("seed", None)

        if colors < 2:
            raise ValueError(f"colors должен быть не меньше 2, получено {colors}")

        if method == "median_cut":
            img = Image.open(input_data).convert("RGB")
            img.quantize(colors=colors, method=Image.MEDIANCUT).convert("RGB").save(output_data)
            return

        if method != "kmeans":
            raise ValueError(
                f"Неизвестный method={method!r}, ожидается 'median_cut' или 'kmeans'"
            )

        rng = np.random.default_rng(seed)
        array = load_rgb(input_data)
        pixels_array = np.ascontiguousarray(array.reshape(-1, 3), dtype=np.float64)

        taken = min(sample_size, pixels_array.shape[0])
        if colors > taken:
            raise ValueError(
                f"colors={colors} больше доступной выборки ({taken} пикселей): "
                f"увеличьте sample_size или уменьшите colors"
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

        save_rgb(centers_array[labels_array].reshape(array.shape), output_data)
