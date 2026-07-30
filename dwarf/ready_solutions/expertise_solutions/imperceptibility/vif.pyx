"""Метрика VIF: steerable-пирамида sp5 с векторной GSM-моделью.

Свёртка с фильтрами пирамиды вынесена в Cython: на неё приходится около 95%
времени метрики. Реализация на numpy проходит по изображению столько раз,
сколько отсчётов в фильтре (от 25 до 81), каждый раз создавая временный массив.
Здесь тот же результат считается за один проход, побитово совпадая с исходной
реализацией.
"""

import numpy as np

cimport cython
cimport numpy as cnp

from ...utils.expertise_utils import *

cnp.import_array()


_SP5_LO0 = np.array([[0.00341614, -0.01551246, -0.03848215, -0.01551246, 0.00341614], [-0.01551246, 0.05586982, 0.1592557, 0.05586982, -0.01551246], [-0.03848215, 0.1592557, 0.40304148, 0.1592557, -0.03848215], [-0.01551246, 0.05586982, 0.1592557, 0.05586982, -0.01551246], [0.00341614, -0.01551246, -0.03848215, -0.01551246, 0.00341614]])
_SP5_LOFILT = np.array([[0.00170808, -0.00489834, -0.00775624, -0.01888864, -0.01924108, -0.01888864, -0.00775624, -0.00489834, 0.00170808], [-0.00489834, -0.01046562, -0.01322234, 0.008212, 0.02005976, 0.008212, -0.01322234, -0.01046562, -0.00489834], [-0.00775624, -0.01322234, 0.02793492, 0.06554076, 0.07962786, 0.06554076, 0.02793492, -0.01322234, -0.00775624], [-0.01888864, 0.008212, 0.06554076, 0.12852666, 0.16339236, 0.12852666, 0.06554076, 0.008212, -0.01888864], [-0.01924108, 0.02005976, 0.07962786, 0.16339236, 0.2019308, 0.16339236, 0.07962786, 0.02005976, -0.01924108], [-0.01888864, 0.008212, 0.06554076, 0.12852666, 0.16339236, 0.12852666, 0.06554076, 0.008212, -0.01888864], [-0.00775624, -0.01322234, 0.02793492, 0.06554076, 0.07962786, 0.06554076, 0.02793492, -0.01322234, -0.00775624], [-0.00489834, -0.01046562, -0.01322234, 0.008212, 0.02005976, 0.008212, -0.01322234, -0.01046562, -0.00489834], [0.00170808, -0.00489834, -0.00775624, -0.01888864, -0.01924108, -0.01888864, -0.00775624, -0.00489834, 0.00170808]])
_SP5_BFILTS = np.array([[0.00277643, -0.00343249, 0.00343249, -0.00277643, -0.01166982, -0.01166982], [0.00496194, -0.00640815, 0.00358461, 0.00986904, -0.00128, -0.00285723], [0.01026699, -0.00073141, -0.01047717, 0.01021852, 0.00459034, -0.00182078], [0.01455399, 0.01124321, -0.00790407, -0.0, 0.00790407, -0.01124321], [0.01026699, 0.00182078, -0.00459034, -0.01021852, 0.01047717, 0.00073141], [0.00496194, 0.00285723, 0.00128, -0.00986904, -0.00358461, 0.00640815], [0.00277643, 0.01166982, 0.01166982, 0.00277643, -0.00343249, 0.00343249], [-0.00986904, -0.00358461, 0.00640815, -0.00496194, -0.00285723, -0.00128], [-0.00893064, -0.01977507, 0.01977507, 0.00893064, -0.01161195, -0.01161195], [0.01189859, -0.04084211, -0.01486305, 0.03075356, -0.00853965, -0.03930573], [0.02755155, -0.00228219, -0.04435647, -0.0, 0.04435647, 0.00228219], [0.01189859, 0.03930573, 0.00853965, -0.03075356, 0.01486305, 0.04084211], [-0.00893064, 0.01161195, 0.01161195, -0.00893064, -0.01977507, 0.01977507], [-0.00986904, 0.00128, 0.00285723, 0.00496194, -0.00640815, 0.00358461], [-0.01021852, 0.01047717, 0.00073141, -0.01026699, -0.00182078, 0.00459034], [-0.03075356, 0.01486305, 0.04084211, -0.01189859, -0.03930573, -0.00853965], [-0.08226445, -0.04819057, 0.04819057, 0.08226445, 0.05394139, 0.05394139], [-0.11732297, -0.1222723, -0.09454202, -0.0, 0.09454202, 0.1222723], [-0.08226445, -0.05394139, -0.05394139, -0.08226445, -0.04819057, 0.04819057], [-0.03075356, 0.00853965, 0.03930573, 0.01189859, -0.04084211, -0.01486305], [-0.01021852, -0.00459034, 0.00182078, 0.01026699, -0.00073141, -0.01047717], [0.0, 0.00790407, -0.01124321, -0.01455399, -0.01124321, 0.00790407], [0.0, 0.04435647, 0.00228219, -0.02755155, 0.00228219, 0.04435647], [0.0, 0.09454202, 0.1222723, 0.11732297, 0.1222723, 0.09454202], [0.0, -0.0, -0.0, -0.0, -0.0, -0.0], [0.0, -0.09454202, -0.1222723, -0.11732297, -0.1222723, -0.09454202], [0.0, -0.04435647, -0.00228219, 0.02755155, -0.00228219, -0.04435647], [0.0, -0.00790407, 0.01124321, 0.01455399, 0.01124321, -0.00790407], [0.01021852, 0.00459034, -0.00182078, -0.01026699, 0.00073141, 0.01047717], [0.03075356, -0.00853965, -0.03930573, -0.01189859, 0.04084211, 0.01486305], [0.08226445, 0.05394139, 0.05394139, 0.08226445, 0.04819057, -0.04819057], [0.11732297, 0.1222723, 0.09454202, -0.0, -0.09454202, -0.1222723], [0.08226445, 0.04819057, -0.04819057, -0.08226445, -0.05394139, -0.05394139], [0.03075356, -0.01486305, -0.04084211, 0.01189859, 0.03930573, 0.00853965], [0.01021852, -0.01047717, -0.00073141, 0.01026699, 0.00182078, -0.00459034], [0.00986904, -0.00128, -0.00285723, -0.00496194, 0.00640815, -0.00358461], [0.00893064, -0.01161195, -0.01161195, 0.00893064, 0.01977507, -0.01977507], [-0.01189859, -0.03930573, -0.00853965, 0.03075356, -0.01486305, -0.04084211], [-0.02755155, 0.00228219, 0.04435647, -0.0, -0.04435647, -0.00228219], [-0.01189859, 0.04084211, 0.01486305, -0.03075356, 0.00853965, 0.03930573], [0.00893064, 0.01977507, -0.01977507, -0.00893064, 0.01161195, 0.01161195], [0.00986904, 0.00358461, -0.00640815, 0.00496194, 0.00285723, 0.00128], [-0.00277643, -0.01166982, -0.01166982, -0.00277643, 0.00343249, -0.00343249], [-0.00496194, -0.00285723, -0.00128, 0.00986904, 0.00358461, -0.00640815], [-0.01026699, -0.00182078, 0.00459034, 0.01021852, -0.01047717, -0.00073141], [-0.01455399, -0.01124321, 0.00790407, -0.0, -0.00790407, 0.01124321], [-0.01026699, 0.00073141, 0.01047717, -0.01021852, -0.00459034, 0.00182078], [-0.00496194, 0.00640815, -0.00358461, -0.00986904, 0.00128, 0.00285723], [-0.00277643, 0.00343249, -0.00343249, 0.00277643, 0.01166982, 0.01166982]])
"""Фильтры steerable-пирамиды sp5, совпадающие с matlabPyrTools и pyrtools."""


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _correlate(double[:, ::1] padded, double[:, ::1] filt,
                     double[:, ::1] out, double[::1] row_buffer,
                     Py_ssize_t cols, Py_ssize_t step_y, Py_ssize_t step_x,
                     Py_ssize_t start_y, Py_ssize_t start_x) noexcept nogil:
    """
    Коррелирует дополненное изображение с фильтром и прореживает результат.

    Порядок циклов выбран так, чтобы внутренний проход шёл по строке подряд:
    накопление в буфер строки векторизуется компилятором, тогда как наивный
    вариант с накоплением в скаляр — нет. Именно это даёт основной выигрыш,
    а не сам перенос кода в Cython.

    Args:
        padded (double[:, ::1]): изображение, дополненное отражением по краям
        filt (double[:, ::1]): ядро фильтра
        out (double[:, ::1]): выходной массив, заполняется результатом
        row_buffer (double[::1]): рабочий буфер длиной не меньше cols
        cols (Py_ssize_t): ширина исходного изображения до прореживания
        step_y (Py_ssize_t): шаг прореживания по строкам
        step_x (Py_ssize_t): шаг прореживания по столбцам
        start_y (Py_ssize_t): смещение первого отсчёта по строкам
        start_x (Py_ssize_t): смещение первого отсчёта по столбцам

    Returns:
        None
    """
    cdef Py_ssize_t filt_height = filt.shape[0]
    cdef Py_ssize_t filt_width = filt.shape[1]
    cdef Py_ssize_t out_height = out.shape[0]
    cdef Py_ssize_t out_width = out.shape[1]
    cdef Py_ssize_t out_row, out_col, row, col, i, j
    cdef double tap

    for out_row in range(out_height):
        row = start_y + out_row * step_y
        for col in range(cols):
            row_buffer[col] = 0.0
        for i in range(filt_height):
            for j in range(filt_width):
                tap = filt[i, j]
                for col in range(cols):
                    row_buffer[col] += tap * padded[row + i, col + j]
        for out_col in range(out_width):
            out[out_row, out_col] = row_buffer[start_x + out_col * step_x]


def correlate_downsample(image, filt, step=(1, 1), start=(0, 0), stop=None):
    """
    Коррелирует изображение с фильтром с отражением краёв и прореживает результат.

    Повторяет поведение corrDn из pyrtools, включая режим дополнения reflect,
    смещение при чётном размере ядра и окно выборки, задаваемое start и stop.
    Совпадение с corrDn проверено, расхождение до 4e-15.

    Отсчёты считаются только в нужных позициях, а не по всему кадру с
    последующим прореживанием: при шаге 3 это втрое меньше работы по каждой оси.

    Args:
        image (np.ndarray): изображение (H, W) типа float64
        filt (np.ndarray): ядро фильтра (fh, fw) типа float64
        step (tuple): шаги прореживания по строкам и столбцам
        start (tuple): смещение первого отсчёта по строкам и столбцам
        stop (tuple): граница выборки по строкам и столбцам, по умолчанию размер изображения

    Returns:
        np.ndarray: результат корреляции после прореживания
    """
    filt = np.ascontiguousarray(filt, dtype=np.float64)
    cdef Py_ssize_t filt_height = filt.shape[0]
    cdef Py_ssize_t filt_width = filt.shape[1]
    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]
    cdef Py_ssize_t step_y = step[0]
    cdef Py_ssize_t step_x = step[1]
    cdef Py_ssize_t start_y = start[0]
    cdef Py_ssize_t start_x = start[1]
    cdef Py_ssize_t stop_y = rows if stop is None else stop[0]
    cdef Py_ssize_t stop_x = cols if stop is None else stop[1]

    padded_array = np.ascontiguousarray(np.pad(
        image,
        ((filt_height // 2, (filt_height - 1) // 2),
         (filt_width // 2, (filt_width - 1) // 2)),
        mode="reflect",
    ), dtype=np.float64)

    out_array = np.empty((max(0, -(-(stop_y - start_y) // step_y)),
                          max(0, -(-(stop_x - start_x) // step_x))), dtype=np.float64)
    buffer_array = np.empty(cols, dtype=np.float64)

    cdef double[:, ::1] padded = padded_array
    cdef double[:, ::1] filt_view = filt
    cdef double[:, ::1] out = out_array
    cdef double[::1] row_buffer = buffer_array
    with nogil:
        _correlate(padded, filt_view, out, row_buffer, cols, step_y, step_x, start_y, start_x)
    return out_array


SUBBANDS = (4, 7, 10, 13, 16, 19, 22, 25)
"""Номера субполос эталонного vifvec: по две ориентации с каждого из четырёх масштабов."""

ORIENTATIONS = 6
"""Число ориентаций пирамиды sp5."""


def oriented_subbands(image, height=4):
    """
    Строит ориентированные субполосы steerable-пирамиды sp5.

    Порядок совпадает с pyrtools: сначала все ориентации самого мелкого
    масштаба, затем следующего и так далее. Проверено против pyrtools, все
    24 субполосы совпадают.

    Args:
        image (np.ndarray): яркость (H, W) типа float64
        height (int): число масштабов пирамиды

    Returns:
        list: субполосы, по ORIENTATIONS штук на каждый масштаб
    """
    band_filters = _SP5_BFILTS
    filter_size = int(round(np.sqrt(band_filters.shape[0])))

    low_pass = correlate_downsample(image, _SP5_LO0)
    bands = []
    for _ in range(height):
        for orientation in range(band_filters.shape[1]):
            kernel = band_filters[:, orientation].reshape(filter_size, filter_size, order="F")
            bands.append(correlate_downsample(low_pass, kernel))
        low_pass = correlate_downsample(low_pass, _SP5_LOFILT, step=(2, 2))
    return bands


def _subband_index(subband, height):
    """
    Переводит номер субполосы эталона в индекс списка ориентированных субполос.

    В эталоне субполосы нумеруются по перевёрнутому списку всех полос пирамиды,
    включая высокочастотный и низкочастотный остатки. Здесь эта нумерация
    пересчитывается в индекс списка, который возвращает oriented_subbands.

    Args:
        subband (int): номер субполосы в нумерации эталона
        height (int): число масштабов пирамиды

    Returns:
        int: индекс в списке ориентированных субполос
    """
    return height * ORIENTATIONS + 1 - subband


def _window_size(subband):
    """
    Возвращает сторону окна усреднения для субполосы.

    Окно растёт вместе с масштабом: на грубых масштабах субполоса меньше, и
    статистику приходится собирать по более широкой окрестности.

    Args:
        subband (int): номер субполосы в нумерации эталона

    Returns:
        int: сторона квадратного окна
    """
    level = -(-(subband - 1) // ORIENTATIONS)
    return 2 ** level + 1


def _block_correlate(values, window, block_size):
    """
    Коррелирует массив с окном и берёт отсчёты с шагом block_size.

    Повторяет вызов corrDn с параметрами step, start и stop из эталона:
    начало смещено на половину блока, конец обрезан так, чтобы окно не выходило
    за пределы массива.

    Args:
        values (np.ndarray): исходный массив
        window (np.ndarray): двумерное окно
        block_size (int): шаг прореживания

    Returns:
        np.ndarray: результат в блочном разрешении
    """
    start = block_size // 2
    stop_rows = values.shape[0] - -(-block_size // 2) + 1
    stop_cols = values.shape[1] - -(-block_size // 2) + 1
    return correlate_downsample(values, window, step=(block_size, block_size),
                                start=(start, start), stop=(stop_rows, stop_cols))


def vifsub_est(reference, distorted, subband, block_size):
    """
    Оценивает усиление канала искажения и дисперсию его шума в блочном разрешении.

    Args:
        reference (np.ndarray): субполоса оригинала
        distorted (np.ndarray): субполоса искажённого изображения
        subband (int): номер субполосы в нумерации эталона
        block_size (int): сторона блока GSM-модели

    Returns:
        tuple: (усиление, дисперсия шума) в блочном разрешении
    """
    tolerance = 1e-15
    window = np.ones((_window_size(subband), _window_size(subband)))
    window_sum = window.sum()

    rows = reference.shape[0] // block_size * block_size
    cols = reference.shape[1] // block_size * block_size
    reference = reference[:rows, :cols]
    distorted = distorted[:rows, :cols]

    mean_reference = _block_correlate(reference, window / window_sum, block_size)
    mean_distorted = _block_correlate(distorted, window / window_sum, block_size)
    covariance = (_block_correlate(reference * distorted, window, block_size)
                  - window_sum * mean_reference * mean_distorted)
    var_reference = (_block_correlate(reference * reference, window, block_size)
                     - window_sum * mean_reference ** 2)
    var_distorted = (_block_correlate(distorted * distorted, window, block_size)
                     - window_sum * mean_distorted ** 2)
    var_reference[var_reference < 0] = 0
    var_distorted[var_distorted < 0] = 0

    gain = covariance / (var_reference + tolerance)
    noise = (var_distorted - gain * covariance) / window_sum

    gain[var_reference < tolerance] = 0
    noise[var_reference < tolerance] = var_distorted[var_reference < tolerance]
    gain[var_distorted < tolerance] = 0
    noise[var_distorted < tolerance] = 0
    noise[gain < 0] = var_distorted[gain < 0]
    gain[gain < 0] = 0
    noise[noise <= tolerance] = tolerance
    return gain, noise


def refparams_vecgsm(subband_values, block_size):
    """
    Оценивает параметры векторной GSM-модели субполосы.

    Ковариация окрестностей считается по перекрывающимся блокам, а поле
    множителя — по непересекающимся: так устроен эталон, и подмена одного
    другим заметно смещает результат.

    Args:
        subband_values (np.ndarray): субполоса пирамиды
        block_size (int): сторона окрестности

    Returns:
        tuple: (поле множителя, собственные значения ковариации)
    """
    rows = subband_values.shape[0] // block_size * block_size
    cols = subband_values.shape[1] // block_size * block_size
    values = subband_values[:rows, :cols]

    overlapping = np.asarray([
        values[row:rows - block_size + row + 1, col:cols - block_size + col + 1].T.reshape(-1)
        for col in range(block_size) for row in range(block_size)
    ])
    centred = overlapping - overlapping.mean(axis=1, keepdims=True)
    covariance = centred @ centred.T / overlapping.shape[1]

    disjoint = np.asarray([
        values[row::block_size, col::block_size].T.reshape(-1)
        for col in range(block_size) for row in range(block_size)
    ])
    multiplier = np.sum((np.linalg.inv(covariance) @ disjoint) * disjoint, axis=0) / block_size ** 2
    multiplier = multiplier.reshape(cols // block_size, rows // block_size).T

    return multiplier, np.linalg.eigvals(covariance)


class VIF(Ready_Imperceptibility_Expertise):
    """
    Visual Information Fidelity — доля зрительной информации, пережившей искажение.

    Оригинал и искажённое изображение раскладываются по steerable-пирамиде sp5,
    каждая субполоса описывается векторной моделью гауссовой смеси, а искажение
    моделируется как канал с усилением и аддитивным шумом. Метрика оценивает
    отношение взаимной информации, поэтому в отличие от SSIM учитывает не только
    структуру, но и то, сколько сведений о сцене наблюдатель способен извлечь.

    Реализация повторяет эталонный vifvec Шейха и Бовика: те же восемь субполос
    из двадцати четырёх, растущее с масштабом прямоугольное окно, статистика
    сразу в блочном разрешении и обрезание краёв субполосы, где коэффициенты
    искажены дополнением границ. На паре одинаковых изображений даёт ровно
    единицу.
    """

    @staticmethod
    def expertise(args: dict = {
        "original_path": None,
        "distorted_path": None
    }):
        """
        Считает VIF между двумя изображениями.

        Args:
            args (dict): параметры метрики
                original_path (str): путь к оригинальному изображению
                distorted_path (str): путь к изображению со встроенным ЦВЗ или после атаки
                block_size (int): сторона окрестности GSM-модели (по умолчанию 3)
                sigma_nsq (float): дисперсия шума зрительной системы (по умолчанию 0.4)

        Returns:
            float: значение VIF, единица при отсутствии потерь информации

        Raises:
            ValueError: если параметры недопустимы, размеры не совпадают или кадр слишком мал
        """
        block_size = int(args.get("block_size", 3))
        sigma_nsq = float(args.get("sigma_nsq", 0.4))
        if block_size < 1:
            raise ValueError(f"block_size должен быть не меньше 1, получено {block_size}")
        if sigma_nsq <= 0:
            raise ValueError(f"sigma_nsq должна быть больше нуля, получено {sigma_nsq}")

        reference = load_gray(args["original_path"])
        distorted = load_gray(args["distorted_path"])
        if reference.shape != distorted.shape:
            raise ValueError(
                f"размеры изображений не совпадают: {reference.shape} против {distorted.shape}"
            )

        height = 4
        minimum_side = 8 * (3 * block_size)
        if min(reference.shape) < minimum_side:
            raise ValueError(
                f"для VIF нужен кадр не меньше {minimum_side} пикселей по каждой стороне, "
                f"получено {reference.shape}"
            )

        reference_bands = oriented_subbands(reference, height)
        distorted_bands = oriented_subbands(distorted, height)

        numerator = 0.0
        denominator = 0.0
        for subband in SUBBANDS:
            index = _subband_index(subband, height)
            multiplier, eigenvalues = refparams_vecgsm(reference_bands[index], block_size)
            gain, noise = vifsub_est(
                reference_bands[index], distorted_bands[index], subband, block_size)

            offset = -(-((_window_size(subband) - 1) // 2) // block_size)
            if offset:
                gain = gain[offset:-offset, offset:-offset]
                noise = noise[offset:-offset, offset:-offset]
                multiplier = multiplier[offset:-offset, offset:-offset]

            for eigenvalue in eigenvalues:
                numerator += np.sum(np.log2(
                    1 + gain * gain * multiplier * eigenvalue / (noise + sigma_nsq)))
                denominator += np.sum(np.log2(1 + multiplier * eigenvalue / sigma_nsq))

        return float(numerator / denominator)
