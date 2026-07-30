"""Атака эквализации гистограммы: глобальная или адаптивная (CLAHE)."""

import numpy as np

cimport cython
cimport numpy as cnp
from libc.math cimport floor

from ...utils.attack_utils import *

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _interpolate(unsigned char[:, ::1] channel,
                       double[:, :, ::1] maps,
                       double[:, ::1] output,
                       double tile_height,
                       double tile_width,
                       Py_ssize_t tiles) noexcept nogil:
    """
    Билинейно смешивает отображения четырёх ближайших тайлов CLAHE.

    Веса берутся по расстоянию до центров тайлов: без смешивания на границах
    тайлов остаются заметные ступеньки яркости.

    Уровень пикселя 0..255 переводится в номер корзины гистограммы, поскольку
    при bins меньше 256 прямая индексация вышла бы за пределы maps.

    Реализация идёт одним проходом по пикселям. Векторизованный вариант на
    numpy требует четырёх временных массивов (H, W) типа float64 плюс
    промежуточные, то есть около 6 * H * W * 8 байт пиковой памяти.

    Args:
        channel (unsigned char[:, ::1]): яркостный канал (H, W) со значениями 0..255
        maps (double[:, :, ::1]): отображения тайлов (tiles, tiles, bins)
        output (double[:, ::1]): выходной массив (H, W), заполняется результатом
        tile_height (double): высота тайла в пикселях
        tile_width (double): ширина тайла в пикселях
        tiles (Py_ssize_t): число тайлов по каждой оси

    Returns:
        None
    """
    cdef Py_ssize_t height = channel.shape[0]
    cdef Py_ssize_t width = channel.shape[1]
    cdef Py_ssize_t bins = maps.shape[2]
    cdef Py_ssize_t y, x, y0, y1, x0, x1, level
    cdef double position_y, position_x, weight_y, weight_x, top, bottom

    for y in range(height):
        position_y = (y + 0.5) / tile_height - 0.5
        if position_y < 0.0:
            position_y = 0.0
        elif position_y > tiles - 1:
            position_y = tiles - 1
        y0 = <Py_ssize_t>floor(position_y)
        y1 = y0 + 1
        if y1 > tiles - 1:
            y1 = tiles - 1
        weight_y = position_y - y0

        for x in range(width):
            position_x = (x + 0.5) / tile_width - 0.5
            if position_x < 0.0:
                position_x = 0.0
            elif position_x > tiles - 1:
                position_x = tiles - 1
            x0 = <Py_ssize_t>floor(position_x)
            x1 = x0 + 1
            if x1 > tiles - 1:
                x1 = tiles - 1
            weight_x = position_x - x0

            level = (<Py_ssize_t>channel[y, x] * bins) // 256
            if level > bins - 1:
                level = bins - 1

            top = maps[y0, x0, level] * (1.0 - weight_x) + maps[y0, x1, level] * weight_x
            bottom = maps[y1, x0, level] * (1.0 - weight_x) + maps[y1, x1, level] * weight_x
            output[y, x] = top * (1.0 - weight_y) + bottom * weight_y


def _build_maps(channel, Py_ssize_t tiles, double clip_limit, Py_ssize_t bins):
    """
    Строит отображение уровней яркости для каждого тайла.

    Размер тайла округляется вверх, чтобы правый и нижний края кадра попадали
    в последний тайл, а не оставались вне всех гистограмм.

    Гистограмма тайла обрезается по порогу clip_limit, срезанная масса
    возвращается в корзины целиком: сначала поровну во все, затем остаток от
    деления раздаётся по корзинам с равномерным шагом. Отбрасывание остатка
    незаметно на крупных тайлах, но на мелких он составляет заметную долю
    массы тайла и заметно искажает накопленную сумму.

    После возврата массы накопленная сумма
    делится на полное число отсчётов и масштабируется в диапазон 0..255.

    Нормировка идёт по числу отсчётов, а не по размаху накопленной суммы.
    Вычитание минимума с делением на размах растянуло бы каждый тайл на весь
    диапазон независимо от его содержимого: на мелких тайлах, где порог
    обрезания опускается до одного отсчёта на корзину, это даёт грубую
    переэквализацию и расхождение с эталонной реализацией.

    Args:
        channel (np.ndarray): яркостный канал (H, W) типа uint8
        tiles (Py_ssize_t): число тайлов по каждой оси
        clip_limit (double): множитель ограничения контраста
        bins (Py_ssize_t): число корзин гистограммы

    Returns:
        tuple: (maps, tile_height, tile_width), где maps — массив (tiles, tiles, bins)
    """
    cdef Py_ssize_t height = channel.shape[0]
    cdef Py_ssize_t width = channel.shape[1]
    cdef Py_ssize_t tile_height = (height + tiles - 1) // tiles
    cdef Py_ssize_t tile_width = (width + tiles - 1) // tiles
    cdef Py_ssize_t row, column
    cdef long limit, excess

    maps = np.zeros((tiles, tiles, bins), dtype=np.float64)
    for row in range(tiles):
        for column in range(tiles):
            block = channel[row * tile_height:(row + 1) * tile_height,
                            column * tile_width:(column + 1) * tile_width]
            if block.size == 0:
                maps[row, column] = np.arange(bins, dtype=np.float64) / max(1, bins - 1) * 255.0
                continue
            histogram, _ = np.histogram(block, bins=bins, range=(0, 256))
            limit = max(1, <long>(clip_limit * block.size / bins))
            excess = int(np.maximum(histogram - limit, 0).sum())
            histogram = np.minimum(histogram, limit)
            batch = excess // bins
            residual = excess - batch * bins
            histogram = histogram + batch
            if residual > 0:
                stride = max(1, bins // residual)
                histogram[::stride][:residual] += 1
            cumulative = np.cumsum(histogram).astype(np.float64)
            total = max(1.0, float(cumulative[-1]))
            maps[row, column] = cumulative / total * 255.0

    return maps, float(tile_height), float(tile_width)


class Histogram_Equalization(Ready_Color_Brightness_Attacks):
    """
    Атака эквализации гистограммы.

    Растягивает распределение яркостей на полный диапазон: глобально по всему
    кадру либо адаптивно по тайлам с ограничением контраста (CLAHE). Нелинейно
    перераспределяет уровни яркости и потому опасна для схем встраивания,
    опирающихся на абсолютные значения пикселей.
    """

    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Выравнивает гистограмму изображения и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                method (str): 'global' или 'clahe' (по умолчанию 'global')
                tiles (int): число тайлов по каждой оси для CLAHE, не меньше 2 (по умолчанию 8)
                clip_limit (float): множитель ограничения контраста для CLAHE (по умолчанию 2.0)
                bins (int): число корзин гистограммы для CLAHE, 2..256 (по умолчанию 256)

        Returns:
            None

        Raises:
            ValueError: если method неизвестен, tiles меньше 2 или bins вне диапазона 2..256
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        method = args.get("method", "global")
        cdef Py_ssize_t tiles = int(args.get("tiles", 8))
        cdef double clip_limit = float(args.get("clip_limit", 2.0))
        cdef Py_ssize_t bins = int(args.get("bins", 256))
        cdef unsigned char[:, ::1] luma
        cdef double[:, :, ::1] maps_view
        cdef double[:, ::1] output_view
        cdef double tile_height, tile_width

        if method == "global":
            img = Image.open(input_data).convert("RGB")
            ImageOps.equalize(img).save(output_data)
            return

        if method != "clahe":
            raise ValueError(
                f"Неизвестный method={method!r}, ожидается 'global' или 'clahe'"
            )
        if tiles < 2:
            raise ValueError(f"tiles должен быть не меньше 2, получено {tiles}")
        if not 2 <= bins <= 256:
            raise ValueError(f"bins должен быть в диапазоне 2..256, получено {bins}")

        img = Image.open(input_data).convert("RGB")
        ycbcr = np.asarray(img.convert("YCbCr")).astype(np.float64)
        luma_array = np.ascontiguousarray(ycbcr[:, :, 0].astype(np.uint8))
        maps, height_step, width_step = _build_maps(luma_array, tiles, clip_limit, bins)
        tile_height = height_step
        tile_width = width_step

        equalized = np.empty(luma_array.shape, dtype=np.float64)
        luma = luma_array
        maps_view = np.ascontiguousarray(maps)
        output_view = equalized
        with nogil:
            _interpolate(luma, maps_view, output_view, tile_height, tile_width, tiles)

        ycbcr[:, :, 0] = equalized
        result = Image.fromarray(np.clip(ycbcr, 0, 255).round().astype(np.uint8), "YCbCr")
        result.convert("RGB").save(output_data)
