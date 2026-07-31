import numpy as np
import warnings

from libc.math cimport sqrt, fabs, floor
from cython.parallel cimport prange  # type: ignore

from ...utils.embedding_utils import *


# Внутренние C-константы
cdef enum:
    N = 8  # Сторона блока
    NN = 64  # N * N, число элементов блока
    MAX_SWEEPS = 30  # Максимум свёрток метода Якоби


cdef void svd_jacobi(double *A, double *V, double *sv, bint want_v) noexcept nogil:
    """
    Односторонний (правосторонний) метод Якоби.
    Последовательно применяет к парам столбцов (p, q) вращения Гивенса,
    делающие их ортогональными. Когда все столбцы попарно ортогональны, A = U*S,
    а произведение всех вращений даёт V.

    Args:
        A (double*): входная матрица N x N.
        V (double*): буфер под правые сингулярные векторы.
        sv (double*): буфер под сингулярные числа.
        want_v (bint): флаг, накапливать ли V (по сути - встраивание или извлечение).
    """
    cdef int p, q, i, sweep # p, q - номера пары обрабатываемых столбцов, i - строка, sweep - номер прохода
    cdef double alpha, beta, gamma, zeta, t, c, s, ap, aq # alpha, beta - квадраты длин столбцов p и q
        # gamma - их скалярное произведение
        # zeta, t, c, s - параметры поворота
        # ap, aq - временные копии элементов
    cdef double off   # Мера неортогональности, чем меньше тем лучше, критерий остановки внешнего цикла

    # V инициализируем единичной матрицей
    # Каждый последующий поворот будет домножать её справа
    # К концу она окажется произведением всех выполненных поворотов
    if want_v:
        for i in range(NN):
            V[i] = 0.0
        for i in range(N):
            V[i * N + i] = 1.0

    # Цикл свёрток (sqeep)
    for sweep in range(MAX_SWEEPS):
        off = 0.0

        # Одна свёртка - проход по всем парам столбцов
        for p in range(N - 1):
            for q in range(p + 1, N):

                # Элементы грамовой подматрицы 2x2
                alpha = 0.0
                beta = 0.0
                gamma = 0.0
                for i in range(N):
                    ap = A[i * N + p] # элемент (i, p)
                    aq = A[i * N + q] # элемент (i, q)
                    alpha += ap * ap
                    beta += aq * aq
                    gamma += ap * aq

                # Cтолбцы уже строго ортогональны (скалярное произведение равно 0) - вращать нечего
                if gamma == 0.0:
                    continue

                off += gamma * gamma / (alpha * beta + 1e-300) # Накопление меры неортогональности

                # Пропуск пренебрежимо малых углов
                if fabs(gamma) <= 1e-15 * sqrt(alpha * beta):
                    continue

                # Параметры вращения Гивенса
                zeta = (beta - alpha) / (2.0 * gamma)
                if zeta >= 0.0:
                    t = 1.0 / (zeta + sqrt(1.0 + zeta * zeta))
                else:
                    t = -1.0 / (-zeta + sqrt(1.0 + zeta * zeta))
                c = 1.0 / sqrt(1.0 + t * t)
                s = c * t

                # Поворот пары столбцов p и q матрицы A
                for i in range(N):
                    ap = A[i * N + p]
                    aq = A[i * N + q]
                    A[i * N + p] = c * ap - s * aq
                    A[i * N + q] = s * ap + c * aq

                # Это же вращение накапливаем в V
                if want_v:
                    for i in range(N):
                        ap = V[i * N + p]
                        aq = V[i * N + q]
                        V[i * N + p] = c * ap - s * aq
                        V[i * N + q] = s * ap + c * aq

        # Все пары столбцов ортогональны с высокой точностью
        # Дальнейшие повороты имеют мало смысла
        if off <= 1e-30:
            break

    # Сингулярные числа - длины столбцов сошедшейся матрицы
    for q in range(N):
        alpha = 0.0
        for i in range(N):
            alpha += A[i * N + q] * A[i * N + q]
        sv[q] = sqrt(alpha)


cdef inline void top2(double *sv, int *idx1, double *s1, double *s2) noexcept nogil:
    """
    Поиск 2 наибольших сингулярных чисел (с определением позиции наибольшего).

    Args:
        sv (double*): сингулярные числа в произвольном порядке,
            результат работы svd_jacobi.
        idx1 (int*): позиция наибольшего числа в массиве sv.
        s1 (double*): наибольшее сингулярное число.
        s2 (double*): второе по величине сингулярное число.
    """
    cdef int i, k1 = 0
    cdef double m1 = sv[0]
    cdef double m2 = -1.0 # Сингулярные числа всегда неотрицательные

    # Нахождение 2 максимальных чисел, обновление номера позиции наибольшего
    for i in range(1, N):
        if sv[i] > m1:
            m2 = m1
            m1 = sv[i]
            k1 = i
        elif sv[i] > m2:
            m2 = sv[i]

    idx1[0] = k1
    s1[0] = m1
    s2[0] = m2


cdef inline double qim_embed(double sigma, int bit, double delta, double sigma2) noexcept nogil:
    """
    Непосредственное встраивание бита ЦВЗ методом QIM

    Кодовая книга бита b:  C_b = { Δ * (k + off_b) },  off_b = 0.25 + 0.5*b, k - целое.

        бит 0 -> точки 0.25*Δ, 1.25*Δ, 2.25*Δ ...
        бит 1 -> точки 0.75*Δ, 1.75*Δ, 2.75*Δ ...

    Выбирается ближайшая точка нужной книги, поэтому искажение не превышает Δ/2,
    а запас до границы решающего правила (детектор режет по 0.0 и 0.5 внутри ячейки) составляет Δ/4.

    Дополнительно гарантируется, что результат останется наибольшим сингулярным
    числом блока, иначе при извлечении детектор прочитает другое, немодифицированное число.
    Коррекция выполняется сдвигом на целое число Δ, что сохраняет значение бита.

    Args:
        sigma (double): текущее старшее сингулярное число блока (неотрицательное).
        bit (int): встраиваемый бит ЦВЗ.
        delta (double): шаг квантования.
            Должен совпадать со значением, переданным qim_extract().
        sigma2 (double): второе по величине сингулярное число блока.

    Returns:
        out (double): новое значение старшего сингулярного числа - точка кодовой
            книги текущего бита, строго положительная и не меньшая sigma2
    """
    cdef double off = 0.25 + 0.5 * bit # Выбор кодовой книги для бита
    cdef double k = floor(sigma / delta - off + 0.5) # Номер ближайшей точки
    cdef double out = delta * (k + off) # Реальное значение сингулярного числа

    # Для извлечения нельзя, чтобы out было меньше немодифицированного sigma2
    while out < sigma2 or out <= 0.0:
        out += delta # Остаёмся в той же кодовой книге

    return out


cdef inline int qim_extract(double sigma, double delta) noexcept nogil:
    """
    Извлечение бита из старшего сингулярного числа методом QIM.

    Анализируется дробная часть sigma/Δ - положение значения внутри ячейки
    квантования:

        [0, 0.5) -> бит 0   (центр кодовой книги 0.25)
        [0.5, 1) -> бит 1   (центр кодовой книги 0.75)

    Границы решения проходят ровно посередине между точками кодовых книг,
    поэтому бит переживает любое возмущение sigma по модулю меньше Δ/4.

    Args:
        sigma (double): наблюдаемое старшее сингулярное число блока.
        delta (double): шаг квантования.
            Должен совпадать со значением, переданным qim_embed().

    Returns:
        (int): извлечённый бит ЦВЗ.
    """
    cdef double r = sigma / delta # Сколько шагов квантования укладывается в наблюдаемое сингулярное число
    cdef double f = r - floor(r) # Дробная часть - положение между точками
    if f >= 0.5:
        return 1
    return 0


cdef void embed_block(double[:, ::1] img, int by, int bx, int bit,
                       double delta) noexcept nogil:
    """
    Встраивание одного бита ЦВЗ в один блок 8x8 изображения.

    Блок копируется в локальный буфер, раскладывается методом Якоби, старшее
    сингулярное число квантуется под заданный бит, после чего результат вносится
    обратно в изображение обновлением ранга 1 - без полной пересборки U*S'*V^T.
    Вырожденные (полностью чёрные) блоки пропускаются.
    
    Args:
        img (2d-memoryview float64, C-совместимый): полное изображение.
        by (int): номер блока по вертикали.
        bx (int): номер блока по горизонтали.
        bit (int): встраиваемый бит ЦВЗ.
        delta (double): шаг квантования.
            Должен совпадать со значением, переданным qim_embed().
    """
    cdef double A[NN] # Блок до применения преобразования Якоби
    cdef double V[NN] # Правые сингулярные векторы (по столбцам)
    cdef double sv[N] # Сингулярные числа

    cdef int i, j, i1
    # Координаты левого верхнего угла блока
    cdef int r0 = by * N
    cdef int c0 = bx * N
    cdef double s1, s2, s_new, scale

    # Копирование блока в C-буфер
    for i in range(N):
        for j in range(N):
            A[i * N + j] = img[r0 + i, c0 + j]

    svd_jacobi(A, V, sv, True) # Метод Якоби
    top2(sv, &i1, &s1, &s2) # Нахождение 2 старших сингулярных чисел (и позиции наибольшего)

    # Отсечение вырожденных блоков
    if s1 < 1e-12:
        return

    s_new = qim_embed(s1, bit, delta, s2) # Новое значение наибольшего сингулярного числа 

    scale = (s_new - s1) / s1 # Подсчёт поправки к блоку
    for i in range(N):
        for j in range(N):
            img[r0 + i, c0 + j] += scale * A[i * N + i1] * V[j * N + i1] # Непосредственная корректировка пикселя


cdef int extract_block(double[:, ::1] img, int by, int bx,
                        double delta) noexcept nogil:
    """
    Извлечение одного бита ЦВЗ из блока 8x8.

    Ни оригинальное изображение, ни матрицы U и V детектору не требуются,
    единственный разделяемый со встраиванием параметр - Δ.

    Args:
        img (2d-memoryview float64, C-совместимый): полное изображение.
        by (int): номер блока по вертикали (не пиксельная координата)
        bx (int): номер блока по горизонтали (не пиксельная координата)
        delta (double): шаг квантования, обязан совпадать с использованным
            при встраивании

    Returns:
        (int): извлечённый бит ЦВЗ.
    """
    cdef double A[NN] # Блок до применения преобразования Якоби
    cdef double sv[N] # Сингулярные числа

    cdef int i, j
    cdef int r0 = by * N
    cdef int c0 = bx * N
    cdef double s1 = 0.0

    # Копирование блока в C-буфер
    for i in range(N):
        for j in range(N):
            A[i * N + j] = img[r0 + i, c0 + j]

    svd_jacobi(A, NULL, sv, False) # Метод Якоби, но без накопления V

    # Нахождение старшего сингулярного числа
    for i in range(N):
        if sv[i] > s1:
            s1 = sv[i]

    return qim_extract(s1, delta) # Извлечение бита ЦВЗ из старшего сингулярного числа


cdef int normalize_redundancy(int redundancy, str func_name):
    """
    Приведение redundancy к нечётному значению.
    При чётном redundancy ровно половина голосов не даёт большинства, и ничья
    в голосовании разрешается строкой votes * 2 > redundancy в пользу нуля.
    На практике это смещает результат.

    Args:
        redundancy (int): запрошенное число копий каждого бита
        func_name (str): имя вызывающей функции, попадает в текст предупреждения

    Returns:
        redundancy (int): redundancy, если оно нечётное, иначе redundancy - 1
    """
    if redundancy % 2 == 0:
        warnings.warn(
            f"{func_name}: redundancy={redundancy} чётное, мажоритарное голосование "
            f"даёт ничью на половине голосов и смещает результат в пользу нуля. "
            f"Значение понижено до {redundancy - 1}. Одно и то же значение redundancy "
            f"должно передаваться и при встраивании, и при извлечении.",
            UserWarning,
            stacklevel=1,
        )
        return redundancy - 1
    return redundancy


def embed_watermark_svd(double[:, ::1] image,
                        int[::1] watermark,
                        double margin=40.0,
                        int redundancy=1):
    """
    Встраивание ЦВЗ во всё изображение.

    Изображение разбивается на блоки 8x8, каждый блок несёт 1 бит ЦВЗ.
    Копии бита i при redundancy > 1 попадают в блоки i, i+L, i+2L, ..., где L - длина ЦВЗ.

    Args:
        image (2d-memoryview float64, C-совместимый): канал яркости изображения Y.
        watermark (1d-memoryview int32, C-совместимый): биты ЦВЗ.
        margin (double): шаг квантования. Больше - устойчивее, но заметнее.
        redundancy (int): сколько блоков тратится на один бит.

    Returns:
        out_np (2d-array float64, C-совместимый): изображение со встроенным ЦВЗ.
    """
    cdef Py_ssize_t H = image.shape[0]
    cdef Py_ssize_t W = image.shape[1]
    cdef Py_ssize_t L = watermark.shape[0] # Длина ЦВЗ

    if margin <= 0.0:
        raise ValueError("margin (шаг квантования) должен быть положительным.")
    if redundancy < 1:
        raise ValueError("redundancy должен быть >= 1.")

    redundancy = normalize_redundancy(redundancy, "embed_watermark_svd") # Приведение redundancy к нечётному

    cdef Py_ssize_t nb_h = H // N
    cdef Py_ssize_t nb_w = W // N
    cdef Py_ssize_t capacity = nb_h * nb_w # Имеющееся количество блоков
    cdef Py_ssize_t used = L * redundancy # Требуемое количество блоков

    if used > capacity:
        raise ValueError(
            f"Не хватает ёмкости: нужно {used} блоков, доступно {capacity}. "
            f"Уменьшите длину ЦВЗ или redundancy."
        )

    out_np = np.array(image, dtype=np.float64, order='C')
    cdef double[:, ::1] out = out_np # Работа с memoryview

    cdef Py_ssize_t b
    cdef int nbw = <int>nb_w

    # Параллельная обработка блоков
    for b in prange(used, nogil=True, schedule='static'):
        embed_block(out,
                     <int>(b // nbw), <int>(b % nbw), # Определение координат блока
                     watermark[b % L], # Расикдывание бит ЦВЗ в соответствии с redundancy
                     margin)

    return out_np


def extract_watermark_svd(double[:, ::1] image,
                          int wm_length,
                          double margin=40.0,
                          int redundancy=1):
    """
    Извлечение ЦВЗ из всего изображения.

    Args:
        image (2d-memoryview float64, C-совместимый): канал яркости изображения Y.
        wm_length (int): длина ЦВЗ.
        margin (double): шаг квантования.
            Должен совпадать со значением margin при встраивании.
        redundancy (int): число копий каждого бита.
            Должен совпадать со значением redundancy при встраивании.

    Returns:
        votes (1d-array int32): извлечённые биты ЦВЗ. При redundancy = 1
            это сырые биты блоков без обработки, при redundancy > 1
            - результат мажоритарного голосования по копиям.
    """
    cdef Py_ssize_t H = image.shape[0]
    cdef Py_ssize_t W = image.shape[1]
    cdef Py_ssize_t L = wm_length # Длина ЦВЗ

    if margin <= 0.0:
        raise ValueError("margin (шаг квантования) должен быть положительным.")
    if redundancy < 1:
        raise ValueError("redundancy должен быть >= 1.")
    
    redundancy = normalize_redundancy(redundancy, "embed_watermark_svd") # Приведение redundancy к нечётному

    cdef Py_ssize_t nb_h = H // N
    cdef Py_ssize_t nb_w = W // N
    cdef Py_ssize_t used = L * redundancy # Количество блоков для извлечения

    if used > nb_h * nb_w:
        raise ValueError("wm_length * redundancy превышает число блоков 8x8.")

    # Буфер для сырых битов (до голосования)
    raw_np = np.empty(used, dtype=np.int32)
    cdef int[::1] raw = raw_np

    cdef Py_ssize_t b
    cdef int nbw = <int>nb_w

    # Параллельная обработка блоков
    for b in prange(used, nogil=True, schedule='static'):
        raw[b] = extract_block(image,
                                <int>(b // nbw), <int>(b % nbw), # Определение координат блока
                                margin)

    # Вывод сырых битов
    if redundancy == 1:
        return raw_np

    # Механизм голосования
    votes = raw_np.reshape(redundancy, L).sum(axis=0)
    return (votes * 2 > redundancy).astype(np.int32)


def capacity_bits(int height, int width, int redundancy=1):
    """
    Максимальная длина ЦВЗ (в битах) для изображения данного размера.
    """
    if redundancy < 1:
        raise ValueError("redundancy должен быть >= 1.")
    return (height // N) * (width // N) // redundancy


class SVD(Ready_Frequency_Embeddings):
    """
    Встраивание ЦВЗ в сингулярные числа блоков 8x8.

    Каждый блок раскладывается односторонним методом Якоби, бит кодируется
    квантованием (QIM) наибольшего сингулярного числа. Сингулярные числа
    устойчивее отдельных коэффициентов преобразования: они меняются плавно
    при слабых искажениях блока. При redundancy > 1 бит дублируется по
    блокам и восстанавливается мажоритарным голосованием.
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
                margin (float): шаг квантования; больше - устойчивее, но заметнее (по умолчанию 40.0)
                redundancy (int): сколько блоков тратится на один бит (по умолчанию 1)

        Returns:
            None
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        watermark = bits_to_array(args["watermark_bits"])
        margin = float(args.get("margin", 40.0))
        redundancy = int(args.get("redundancy", 1))

        luma, chroma = load_luma(input_data)
        save_luma(
            embed_watermark_svd(luma, watermark, margin, redundancy),
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

        Args:
            args (dict): параметры извлечения
                input_data (str): путь к изображению с ЦВЗ
                num_bits (int): длина извлекаемого ЦВЗ в битах
                margin (float): шаг квантования (по умолчанию 40.0).
                    Должен совпадать со значением при встраивании.
                redundancy (int): число копий каждого бита (по умолчанию 1).
                    Должен совпадать со значением при встраивании.

        Returns:
            str: строка из '0' и '1' длины num_bits
        """
        input_data = args["input_data"]
        num_bits = int(args["num_bits"])
        margin = float(args.get("margin", 40.0))
        redundancy = int(args.get("redundancy", 1))

        luma, _ = load_luma(input_data)
        return array_to_bits(
            extract_watermark_svd(luma, num_bits, margin, redundancy)
        )

    @staticmethod
    def capacity(args: dict = {
        "input_data": None
    }):
        """
        Максимальная длина ЦВЗ для изображения с учётом избыточности.

        Args:
            args (dict): параметры расчёта
                input_data (str): путь к изображению
                redundancy (int): число копий каждого бита (по умолчанию 1)

        Returns:
            int: ёмкость в битах
        """
        luma, _ = load_luma(args["input_data"])
        return capacity_bits(luma.shape[0], luma.shape[1], int(args.get("redundancy", 1)))
