import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, atan2, log, M_PI
from ...utils.embedding_utils import *

cnp.import_array()

BARKER13 = np.array([+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1],
                    dtype=np.float64) # Маркер для извлечения после поворота

cdef cnp.ndarray build_cell_map(int H, int W, int n_rho, int n_theta,
                                 double r1, double r2):
    """
    Построение карты ячеек лог-полярной сетки после преобразования Фурье.
    Для каждого отсчёта в частотной плоскости определяет, попадает ли он в рабочее кольцо,
    если да - в какую ячейку (подкольцо k, сектор m), если нет - записывает -1.

    Args:
        H (int): высота изображения в пикселях
        W (int): ширина изображения в пикселях
        n_rho (int): число подколец, равномерных по log(r)
        n_theta (int): число угловых секторов на диапазон [0, pi)
        r1 (double): внутренняя граница кольца, периодов на пиксель (0 < r1 < r2)
        r2 (double): внешняя граница кольца, периодов на пиксель (r2 < 0.5)

    Returns:
        out (2d-array int32, C-совместимый): итоговая карта
    """
    cdef cnp.ndarray[cnp.int32_t, ndim=2, mode='c'] out = np.empty((H, W), dtype=np.int32) # Будущая карта в LPM
    cdef int[:, ::1] cell = out # Работа с memoryview

    cdef double log_r1 = log(r1) # Заранее вычисляем логарифм внутреннего радиуса
    # Шаги по радиусу и по углу (обратные к ним величины - чтобы умножать, а не делить)
    cdef double inv_drho = n_rho / (log(r2) - log_r1)
    cdef double inv_dtheta = n_theta / M_PI
    # Нормировка частот в периоды на пиксель
    cdef double inv_H = 1.0 / H
    cdef double inv_W = 1.0 / W

    cdef int i, j, k, m
    cdef int h_half = H // 2
    cdef int w_half = W // 2
    cdef double u, v, r, th, rsq
    cdef double r1sq = r1 * r1
    cdef double r2sq = r2 * r2

    with nogil: # Выполнение цикла с компиляцией в чистый C
        for i in range(H):
            u = <double>(i if i <= h_half else i - H) * inv_H # Частота строки
            for j in range(W):
                v = <double>(j if j <= w_half else j - W) * inv_W # Частота столбца

                # Запись -1 в карту для не интересующих нас отсчётов
                rsq = u * u + v * v
                if rsq < r1sq or rsq >= r2sq:
                    cell[i, j] = -1
                    continue

                r = sqrt(rsq)

                # Переход к логарифму от радиуса (преобразование Меллина)
                k = <int>((log(r) - log_r1) * inv_drho)
                if k < 0:
                    k = 0
                elif k >= n_rho:
                    k = n_rho - 1

                th = atan2(v, u) # Угловой индекс
                # Свёртка углов в диапазон [0, pi) для одинакового усиления центрально противоположных точек
                if th < 0.0:
                    th += M_PI
                if th >= M_PI:
                    th -= M_PI
                m = <int>(th * inv_dtheta) # Номер сектора
                if m < 0:
                    m = 0
                elif m >= n_theta:
                    m = n_theta - 1

                cell[i, j] = k * n_theta + m # Запись в карту
    return out


cdef void accumulate(double complex[:, ::1] F, int[:, ::1] cell,
                      double[::1] sum_log, cnp.int64_t[::1] count) noexcept nogil:
    """
    Накопление статистик по ячейкам.
    Для каждого отсчёта спектра, попавшего в рабочее кольцо, добавляет
    ln(1 + |F|) к сумме своей ячейки и увеличивает её счётчик.
    Отсчёты с cell[i, j] = -1 (вне кольца) пропускаются.

    Args:
        F (2d-memoryview complex128, C-совместимый): спектр изображения в
            "несдвинутой" раскладке np.fft.fft2
        cell (2d-memoryview int32, C-совместимый): карта ячеек той же формы,
            результат build_cell_map()
        sum_log (1d-memoryview float64, C-совместимый): накопитель сумм для отсчётов внутри кольца
        count (1d-memoryview int64, C-совместимый): накопитель числа отсчётов внутри кольца
    """
    cdef Py_ssize_t H = F.shape[0], W = F.shape[1], i, j
    cdef int c # Номер ячейки
    cdef double re, im # Части комплексного числа

    for i in range(H):
        for j in range(W):
            c = cell[i, j] # Чтение параметров ячейки
            if c < 0: # Если вне кольца - идём дальше
                continue
            re = F[i, j].real
            im = F[i, j].imag
            sum_log[c] += log(1.0 + sqrt(re * re + im * im)) # Добавление к очередному отчёту
            count[c] += 1 # Ещё один отчёт попал в текущую ячейку


cdef void apply_gain(double complex[:, ::1] F, int[:, ::1] cell,
                     double[::1] gain) noexcept nogil:
    """
    Умножение коэффициентов спектра на коэффициент усиления своей ячейки.

    Args:
        F (2d-memoryview complex128, C-совместимый): спектр в "несдвинутой"
            раскладке np.fft.fft2
        cell (2d-memoryview int32, C-совместимый): карта ячеек той же формы,
            результат build_cell_map()
        gain (1d-memoryview float64, C-совместимый): коэффициент усиления
            для каждой ячейки
    """
    cdef Py_ssize_t H = F.shape[0], W = F.shape[1], i, j
    cdef int c # Номер ячейки
    cdef double g # Коэффициент усиления

    for i in range(H):
        for j in range(W):
            c = cell[i, j] # Чтение параметров ячейки
            if c < 0: # Если вне кольца - идём дальше
                continue
            g = gain[c] # Получение коэффициента и собственно усиление
            F[i, j].real = F[i, j].real * g
            F[i, j].imag = F[i, j].imag * g


cdef cnp.ndarray cell_mean_sync(double complex[:, ::1] F, int[:, ::1] cell,
                             int n_rho, int n_theta):
    """
    Подсчёт средних ln(1 + |F|) по ячейкам лог-полярной сетки, обработка пустых ячеек.

    Args:
        F (2d-memoryview complex128, C-совместимый): спектр в "несдвинутой"
            раскладке np.fft.fft2
        cell (2d-memoryview int32, C-совместимый): карта ячеек той же формы,
            результат build_cell_map()
        n_rho (int): число подколец
        n_theta (int): число угловых секторов

    Returns:
        S (2d-array float64, C-совместимый, форма (n_rho, n_theta)): среднее
            ln(1 + |F|) по каждой ячейке, для пустых ячеек - средний уровень их подкольца
    """
    cdef int n_cells = n_rho * n_theta # Общее число ячеек
    # Буферы накопления
    cdef cnp.ndarray[cnp.float64_t, ndim=1] s = np.zeros(n_cells, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] c = np.zeros(n_cells, dtype=np.int64)
    # Работа с memoryview
    cdef double[::1] s_mv = s
    cdef cnp.int64_t[::1] c_mv = c
    cdef int t

    with nogil: # Выполнение цикла с компиляцией в чистый C
        accumulate(F, cell, s_mv, c_mv) # Подсчёт статистик по ячейкам
        for t in range(n_cells): # Перевод сумм в средние
            if c_mv[t] > 0:
                s_mv[t] = s_mv[t] / c_mv[t]

    # Распаковка в вид "кольцо х сектор"
    S = s.reshape(n_rho, n_theta)
    C = c.reshape(n_rho, n_theta)
    if (C == 0).any(): # Обработка пустых ячеек
        occ = C > 0
        ring_mean = np.where(occ.any(axis=1),
                             (S * occ).sum(axis=1) / np.maximum(occ.sum(axis=1), 1),
                             0.0)
        S = np.where(occ, S, ring_mean[:, None])
    return np.ascontiguousarray(S)

def embed_watermark_dft(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] image,
                        cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] watermark,
                        double margin=0.40,
                        int n_rho=4,
                        double r_min=0.10,
                        double r_max=0.42,
                        int n_sync=13,
                        double sync_boost=1.5,
                        int n_iter=3,
                        double max_gain=3.0):
    """
    Встраивает ЦВЗ в амплитудный спектр изображения.

    Рабочее кольцо спектра размечается лог-полярной сеткой на подкольца
    (равномерные по log r) и угловые секторы. Секторы объединяются в пары:
    один бит ЦВЗ = одна пара, бит кодируется знаком разности средних
    ln(1 + |F|) внутри пары. Символ дублируется во всех подкольцах.
    Первые n_sync бит заняты кодом Баркера - по нему детектор
    восстанавливает поворот изображения.

    Args:
        image (2d-array float64, C-совместимый): канал яркости изображения Y
        watermark (1d-array int32): ЦВЗ
        margin (double): целевая разность средних ln|F| внутри пары секторов
        n_rho (int): число подколец, т.е. кратность повторения символа вдоль log r
        r_min (double): внутренняя граница рабочего кольца в периодах на пиксель.
            Строго > 0: низкие частоты не трогаем вследствие ухудшения незаметности.
        r_max (double): внешняя граница в периодах на пиксель.
            Строго < 0.5 (частота Найквиста): высокие частоты уничтожаются JPEG и
            сглаживанием, а условие r_max < 0.5 дополнительно гарантирует
            точное сохранение эрмитовой симметрии для точного восстановления ЦВЗ.
        n_sync (int): число символов маркера (кода Баркера).
            Равенство 0 делает схему встраивания неустойчивой к повороту.
        sync_boost (double): во сколько раз амплитуда символов маркера больше margin.
        n_iter (int): число итераций уточнения усиления.
        max_gain (double): ограничение на коэффициент усиления ячейки сверху и снизу.

    Returns:
        watermarked_img (2d-array float64, C-совместимый): изображение с ЦВЗ.
    """
    cdef int H = image.shape[0], W = image.shape[1]
    cdef int L = watermark.shape[0] # Длина ЦВЗ
    cdef int n_sec = n_sync + L # Всего бит (+ длина маркера)
    cdef int n_theta = 2 * n_sec # Секторов вдвое больше
    cdef int n_cells = n_rho * n_theta # Количество ячеек

    if L <= 0:
        raise ValueError("Пустой водяной знак")
    if not (0.0 < r_min < r_max < 0.5):
        raise ValueError("Требуется 0 < r_min < r_max < 0.5")
    if n_rho < 1:
        raise ValueError("n_rho должно быть >= 1")

    # Прямое дискретное преобразование Фурье (numpy)
    cdef cnp.ndarray[cnp.complex128_t, ndim=2, mode='c'] F = \
        np.ascontiguousarray(np.fft.fft2(image), dtype=np.complex128)
    cdef double complex[:, ::1] F_mv = F

    # Построение карты ячеек лог-полярной сетки
    cdef cnp.ndarray[cnp.int32_t, ndim=2, mode='c'] cellm = \
        build_cell_map(H, W, n_rho, n_theta, r_min, r_max)
    cdef int[:, ::1] cell_mv = cellm

    # Вся встраиваемая последовательность: маркер (код Баркера) + ЦВЗ
    sym = np.empty(n_sec, dtype=np.float64)
    if n_sync > 0:
        # Встраиваем маркер перед ЦВЗ, в sync_boost раз "громче" ЦВЗ
        sym[:n_sync] = BARKER13[np.arange(n_sync) % 13] * sync_boost
    sym[n_sync:] = np.where(np.asarray(watermark) > 0, 1.0, -1.0)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] target = margin * sym

    # Итерационное встраивание с усилением коэффициентов спектра
    cdef cnp.ndarray[cnp.float64_t, ndim=1] gain = np.empty(n_cells, dtype=np.float64)
    cdef double[::1] gain_mv = gain
    cdef double lg = log(max_gain)
    cdef int it

    for it in range(n_iter):
        S = cell_mean_sync(F_mv, cell_mv, n_rho, n_theta)
        d = S[:, 0::2] - S[:, 1::2] # Текущая разность пар
        adj = np.clip(0.5 * (target[None, :] - d), -lg, lg)
        G = np.empty((n_rho, n_theta), dtype=np.float64)
        G[:, 0::2] = np.exp(adj) # Чётный сектор пары
        G[:, 1::2] = np.exp(-adj) # Нечётный сектор пары
        gain[:] = G.reshape(-1)
        with nogil:
            apply_gain(F_mv, cell_mv, gain_mv) # Применение усиления

    # Обратное дискретное преобразование Фурье
    return np.ascontiguousarray(np.fft.ifft2(F).real, dtype=np.float64)


def extract_watermark_dft(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] image,
                          int wm_length,
                          int n_rho=4,
                          double r_min=0.10,
                          double r_max=0.42,
                          int n_sync=13,
                          bint search_rotation=True,
                          int oversampling=4,
                          bint return_shift=False):
    """
    Извлекает ЦВЗ.

    Строит ту же лог-полярную сетку, что и при встраивании, вычисляет
    средние ln(1 + |F|) по ячейкам и суммирует их по подкольцам.
    Далее по маркеру оценивается циклический сдвиг секторов (поворот изображения),
    и биты читаются как знак разности внутри каждой пары секторов.

    Args:
        image (2d-array float64, C-совместимый): канал яркости изображения Y
        wm_length (int): длина извлекаемого ЦВЗ.
        n_rho (int): число подколец.
            ОБЯЗАНО совпадать со встраиванием.
        r_min (double): внутренняя граница кольца, периодов на пиксель.
            ОБЯЗАНА совпадать со встраиванием
        r_max (double): внешняя граница кольца, периодов на пиксель.
            ОБЯЗАНА совпадать со встраиванием
        n_sync (int): число символов маркера (кода Бакрера).
            ОБЯЗАНО совпадать со встраиванием.
        search_rotation (bint): True - оценить циклический сдвиг секторов корреляцией с кодом Баркера
            False - считать сдвиг нулевым (чуть надёжнее, если известно, что поворота не было)
        oversampling (int): во сколько раз мельче сетка по углу при поиске сдвига.
        return_shift (bint): вернуть также найденный сдвиг и пик корреляции.

    Returns:
        bits (1d-array int32, длина wm_length): извлечённый ЦВЗ (при return_shift=False)

        При return_shift=True:
            bits (1d-array int32): извлечённый ЦВЗ
            best (int): найденный сдвиг.
            bestc (double): значение пика корреляции.
    """
    cdef int H = image.shape[0], W = image.shape[1]
    cdef int n_sec = n_sync + wm_length # Вся извлекаемая последовательность: маркер (код Баркера) + ЦВЗ
    cdef int OS = oversampling if (search_rotation and n_sync > 0) else 1 # Без поиска поворота мелкая сетка не нужна
    cdef int n_fine = 2 * n_sec * OS

    if OS < 1:
        raise ValueError("oversampling должно быть >= 1")

    # Прямое дискретное преобразование Фурье (numpy)т
    cdef cnp.ndarray[cnp.complex128_t, ndim=2, mode='c'] F = \
        np.ascontiguousarray(np.fft.fft2(image), dtype=np.complex128)
    cdef double complex[:, ::1] F_mv = F

    # Построение карты ячеек лог-полярной сетки
    cdef cnp.ndarray[cnp.int32_t, ndim=2, mode='c'] cellm = \
        build_cell_map(H, W, n_rho, n_fine, r_min, r_max)

    # Суммирование средних по подкольцам
    S = cell_mean_sync(F_mv, cellm, n_rho, n_fine)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] A = np.ascontiguousarray(S.sum(axis=0))
    cdef double[::1] A_mv = A

    cdef cnp.ndarray[cnp.float64_t, ndim=1] bark = \
        np.ascontiguousarray(BARKER13[np.arange(n_sync if n_sync > 0 else 1) % 13])
    cdef double[::1] b_mv = bark
    cdef int tau, best = 0, j, p, m0, m1
    cdef double c, bestc = -1e300, sa, sb

    # Перебор всех сдвиги, для каждого считается совпадение с маркером, побеждает максимум
    # Найденный best затем сдвигает индексы, по которым читаются данные
    if search_rotation and n_sync > 0:
        with nogil:
            for tau in range(n_fine):
                c = 0.0
                for j in range(n_sync):
                    sa = 0.0
                    sb = 0.0
                    m0 = (2 * j) * OS + tau
                    m1 = (2 * j + 1) * OS + tau
                    for p in range(OS):
                        sa += A_mv[(m0 + p) % n_fine]
                        sb += A_mv[(m1 + p) % n_fine]
                    c += b_mv[j] * (sa - sb)
                if c > bestc:
                    bestc = c
                    best = tau

    # Определение битов
    cdef cnp.ndarray[cnp.int32_t, ndim=1] bits = np.empty(wm_length, dtype=np.int32)
    cdef int[::1] bits_mv = bits
    cdef int i

    with nogil:
        for i in range(wm_length):
            sa = 0.0
            sb = 0.0
            m0 = (2 * (n_sync + i)) * OS + best
            m1 = (2 * (n_sync + i) + 1) * OS + best
            for p in range(OS):
                sa += A_mv[(m0 + p) % n_fine]
                sb += A_mv[(m1 + p) % n_fine]
            bits_mv[i] = 1 if sa > sb else 0

    if return_shift:
        return bits, best, bestc
    return bits

class DFT(Ready_Frequency_Embeddings):
    """
    Встраивание ЦВЗ в амплитудный спектр Фурье.

    Рабочее кольцо спектра размечается лог-полярной сеткой; бит кодируется
    знаком разности средних ln(1 + |F|) внутри пары угловых секторов и
    дублируется по подкольцам. Первые n_sync бит занимает код Баркера, по
    которому детектор восстанавливает поворот изображения, - за счёт этого
    схема, в отличие от DCT и DWT, переживает геометрические атаки поворота.
    """

    @staticmethod
    def embedding(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Встраивает ЦВЗ в канал яркости изображения и сохраняет результат.

        Args:
            args (dict): параметры встраивания
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения изображения с ЦВЗ
                watermark_bits (str | Sequence[int]): биты ЦВЗ
                margin (float): целевая разность средних ln|F| внутри пары секторов (по умолчанию 0.40)
                n_rho (int): число подколец, кратность повторения символа вдоль log r (по умолчанию 4)
                r_min (float): внутренняя граница кольца, периодов на пиксель (по умолчанию 0.10)
                r_max (float): внешняя граница кольца, периодов на пиксель (по умолчанию 0.42)
                n_sync (int): число символов маркера, 0 отключает устойчивость к повороту (по умолчанию 13)
                sync_boost (float): во сколько раз амплитуда маркера больше margin (по умолчанию 1.5)
                n_iter (int): число итераций уточнения усиления (по умолчанию 3)
                max_gain (float): ограничение на коэффициент усиления ячейки (по умолчанию 3.0)

        Returns:
            None
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        watermark = bits_to_array(args["watermark_bits"])
        margin = float(args.get("margin", 0.40))
        n_rho = int(args.get("n_rho", 4))
        r_min = float(args.get("r_min", 0.10))
        r_max = float(args.get("r_max", 0.42))
        n_sync = int(args.get("n_sync", 13))
        sync_boost = float(args.get("sync_boost", 1.5))
        n_iter = int(args.get("n_iter", 3))
        max_gain = float(args.get("max_gain", 3.0))

        luma, chroma = load_luma(input_data)
        save_luma(
            embed_watermark_dft(luma, watermark, margin, n_rho, r_min, r_max,
                                n_sync, sync_boost, n_iter, max_gain),
            chroma,
            output_data,
        )

    @staticmethod
    def extraction(args: dict = {
        "input_data": None,
        "num_bits": None
    }):
        """
        Извлекает ЦВЗ из канала яркости изображения.

        Параметры сетки обязаны совпадать со значениями, использованными при
        встраивании: детектор строит ту же лог-полярную разметку.

        Args:
            args (dict): параметры извлечения
                input_data (str): путь к изображению с ЦВЗ
                num_bits (int): длина извлекаемого ЦВЗ в битах
                n_rho (int): число подколец (по умолчанию 4)
                r_min (float): внутренняя граница кольца (по умолчанию 0.10)
                r_max (float): внешняя граница кольца (по умолчанию 0.42)
                n_sync (int): число символов маркера (по умолчанию 13)
                search_rotation (bool): искать циклический сдвиг секторов (по умолчанию True)
                oversampling (int): во сколько раз мельче сетка по углу при поиске сдвига (по умолчанию 4)

        Returns:
            str: строка из '0' и '1' длины num_bits
        """
        input_data = args["input_data"]
        num_bits = int(args["num_bits"])
        n_rho = int(args.get("n_rho", 4))
        r_min = float(args.get("r_min", 0.10))
        r_max = float(args.get("r_max", 0.42))
        n_sync = int(args.get("n_sync", 13))
        search_rotation = bool(args.get("search_rotation", True))
        oversampling = int(args.get("oversampling", 4))

        luma, _ = load_luma(input_data)
        return array_to_bits(
            extract_watermark_dft(luma, num_bits, n_rho, r_min, r_max,
                                  n_sync, search_rotation, oversampling, False)
        )
