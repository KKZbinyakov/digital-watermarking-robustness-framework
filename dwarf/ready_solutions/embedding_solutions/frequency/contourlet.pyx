import numpy as np
cimport numpy as cnp
from libc.math cimport fabs
from ...utils.embedding_utils import *

cnp.import_array()


cdef double H5[5] # Низкочастотный фильтр Бёрта-Адельсона для лапласовой пирамиды
cdef bint FILTERS_INITIALIZED = False # Флаг инициализации фильтра

cdef void init_filters() noexcept nogil:
    """
    Однократная инициализация C-массива коэффициентов фильтра.
    """
    global FILTERS_INITIALIZED
    if FILTERS_INITIALIZED:
        return
    H5[0] = 0.05
    H5[1] = 0.25
    H5[2] = 0.40
    H5[3] = 0.25
    H5[4] = 0.05
    FILTERS_INITIALIZED = True


cdef inline int reflect(int i, int n) noexcept nogil:
    """
    Зеркальное отражение индекса, вышедшего за границы массива.
    Необходимо для подсчёта взвешенной суммы пикселя и его соседей с обеих сторон.
    Пример: ... 2  1 | 0  1  2 ... n-2  n-1 | n-2  n-3 ...

    Args:
        i (int): индекс, возможно выходящий за границы
        n (int): размер массива по данной оси

    Returns:
        (int): индекс в диапазоне [0, n)
    """
    if n == 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i
        if i >= n:
            i = 2 * (n - 1) - i
    return i


# ============================================================================
#  Раздел 1: Лапласова пирамида (многомасштабное разложение)
# ============================================================================

cdef void lowpass2d(double[:, :] src, double[:, :] dst, double[:, :] tmp,
                     double gain) noexcept nogil:
    """
    Двумерная низкочастотная фильтрация ядром Бёрта-Адельсона,
    выполненная как два одномерных прохода.

    gain бывает равен 1.0 (обычное сглаживание, средняя яркость сохраняется)
    и 4.0 (интерполяция после растяжения, из каждых четырёх пикселей новой сетки
    один несёт реальное значение, а три — нули, поэтому средняя яркость падает в 4 раза)

    Args:
        src (2d-memoryview float64, C-совместимый): входная матрица
        dst (2d-memoryview float64, C-совместимый): выходная матрица
        tmp (2d-memoryview float64, C-совместимый): буфер
            для промежуточного результата после горизонтального прохода
        gain (double): общий множитель результата.
    """
    cdef int H = src.shape[0]
    cdef int W = src.shape[1]
    cdef int i, j, t, jj, ii
    cdef double acc # Накопитель суммы

    # Проход 1: свёртка по горизонтали
    for i in range(H):
        for j in range(W):
            acc = 0.0
            for t in range(5): # Пять слагаемых пятиточечного ядра
                jj = reflect(j + t - 2, W) # Отражение индексов, вылезших за края строки
                acc += H5[t] * src[i, jj] # Накапливание суммы
            tmp[i, j] = acc # Запись в буфер - сглаженная по строкам матрица

    # Проход 2: свёртка по вертикали
    for i in range(H):
        for j in range(W):
            acc = 0.0
            for t in range(5):
                ii = reflect(i + t - 2, H)
                # Читаем из буфера - то есть из уже сглаженного по горизонтали изображения.
                acc += H5[t] * tmp[ii, j]
            dst[i, j] = gain * acc # Домножение на усиление


cdef inline void downsample2(double[:, :] src, double[:, :] dst) noexcept nogil:
    """
    Прореживание изображения вдвое по обеим осям.
    Из каждого квадрата 2x2 исходного массива остаётся только левый
    верхний пиксель, остальные три отбрасываются.

    Функция не выполняет предварительного сглаживания. Подавление
    высоких частот перед прореживанием - обязанность вызывающего кода.
    Сначала lowpass2d c gain 1.0, только потом downsample2.

    Args:
        src (2d-memoryview float64, C-совместимый): входная матрица
        dst (2d-memoryview float64, C-совместимый): выходная матрица вдвое меньшего размера
    """
    cdef int Hc = dst.shape[0]
    cdef int Wc = dst.shape[1]
    cdef int i, j
    for i in range(Hc):
        for j in range(Wc):
            dst[i, j] = src[2 * i, 2 * j]


cdef inline void upsample2(double[:, :] src, double[:, :] dst) noexcept nogil:
    """
    Растяжение изображения вдвое по обеим осям с вставкой нулей:
    dst[2i, 2j] = src[i, j], остальные три позиции квадрата 2x2 заполнены нулями.

    Функция записывает только позиции с чётными индексами, поэтому массив dst
    должен быть обнулён вызывающей стороной (обычно через np.zeros).
    
    После данной функции классически следует lowpass2d() с gain = 4.0.

    Args:
        src (2d-memoryview float64, C-совместимый): входная матрица
        dst (2d-memoryview float64, C-совместимый): выходная матрица вдвое большего размера
    """
    cdef int Hc = src.shape[0]
    cdef int Wc = src.shape[1]
    cdef int i, j
    for i in range(Hc):
        for j in range(Wc):
            dst[2 * i, 2 * j] = src[i, j]


cdef lp_analysis_step(double[:, :] x):
    """
    Один уровень анализа лапласовой пирамиды.
    
    Суммарно на выходе (H // 2) * (W // 2) + H * W отсчётов против H * W
    на входе: коэффициент избыточности стремится к 4/3 при увеличении числа уровней.

    Args:
        x (2d-memoryview float64, C-совместимый): входное изображение данного уровня пирамиды,
            H и W должны быть чётными

    Returns:
        coarse (2d-array, C-совместимый): грубое приближение, вход для следующего уровня пирамиды
        band (2d-array, C-совместимый): полосовой остаток, вход для направленного банка фильтров
    """
    cdef int H = x.shape[0]
    cdef int W = x.shape[1]
    cdef int Hc = H // 2
    cdef int Wc = W // 2
    cdef int i, j

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] lp = \
        np.empty((H, W), dtype=np.float64) # Cглаженное изображение перед прореживанием
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] tmp = \
        np.empty((H, W), dtype=np.float64) # Буфер для lowpass2d
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] up = \
        np.zeros((H, W), dtype=np.float64) # Грубый уровень с нулями (для upsample2)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] pred = \
        np.empty((H, W), dtype=np.float64) # Предсказание
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] coarse = \
        np.empty((Hc, Wc), dtype=np.float64) # Выход функции
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] band = \
        np.empty((H, W), dtype=np.float64) # Выход функции

    # Работа с memoryview
    cdef double[:, :] v_lp = lp
    cdef double[:, :] v_tmp = tmp
    cdef double[:, :] v_up = up
    cdef double[:, :] v_pred = pred
    cdef double[:, :] v_coarse = coarse
    cdef double[:, :] v_band = band

    with nogil:
        # Построение грубого уровня
        lowpass2d(x, v_lp, v_tmp, 1.0) # Сглаживание с gain 1.0 убирает высокие частоты
        downsample2(v_lp, v_coarse) # Прореживание

        # Построение предсказания для следующего уровня пирамиды
        upsample2(v_coarse, v_up) # Растяжение со вставкой нулей
        lowpass2d(v_up, v_pred, v_tmp, 4.0) # Сглаживание с gain 4.0 для восставления яркости

        # Сбор остатков (то, что предсказание не смогло воспроизвести: контуры, текстуру, мелкие детали).
        # band идёт в направленный банк фильтров, и именно в его коэффициенты встраивается ЦВЗ.
        for i in range(H):
            for j in range(W):
                v_band[i, j] = x[i, j] - v_pred[i, j] # 

    return coarse, band


cdef lp_synthesis_step(double[:, :] coarse, double[:, :] band):
    """
    Один уровень синтеза лапласовой пирамиды: восстановление изображения
    из грубого приближения и полосового остатка.

    Args:
        coarse (2d-memoryview float64, C-совместимый): грубое приближение данного уровня
        band (2d-memoryview float64, C-совместимый): полосовой остаток данного уровня

    Returns:
        out (2d-array float64, C-совместимый): восстановленное изображение уровнем выше
    """
    cdef int H = band.shape[0]
    cdef int W = band.shape[1]
    cdef int i, j

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] up = \
        np.zeros((H, W), dtype=np.float64) # Грубый уровень с нулями (для upsample2)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] tmp = \
        np.empty((H, W), dtype=np.float64) # Буфер для lowpass2d
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] out = \
        np.empty((H, W), dtype=np.float64) # Выход функции

    # Работа с memoryview
    cdef double[:, :] v_up = up
    cdef double[:, :] v_tmp = tmp
    cdef double[:, :] v_out = out

    with nogil:
        upsample2(coarse, v_up) # Растяжение со вставкой нулей
        lowpass2d(v_up, v_out, v_tmp, 4.0) # Сглаживание с gain 4.0 для восставления яркости
        for i in range(H):
            for j in range(W):
                v_out[i, j] += band[i, j] # Накопление остатка в выход

    return out


# ============================================================================
#  Рзадел 2: Направленный банк фильтров (многонаправленное разложение)
# ============================================================================

# Веса лифтинга

# Веса шага предсказания
cdef double A_PRED = 10.0 / 32.0 # 4 ближних соседа
cdef double B_PRED = -1.0 / 32.0 # 8 дальних соседей
# Веса шага обновления
cdef double A_UPD = 5.0 / 32.0 # 4 ближних соседа
cdef double B_UPD = -0.5 / 32.0 # 8 дальних соседей

# Смещения в решёточных координатах.
# Первые 4 - ближние соседи (вес A_PRED / A_UPD).
# Последние 8 - дальние (вес B_PRED / B_UPD).
cdef int PRED_OFF[12][2]
cdef int UPD_OFF[12][2]

cdef bint OFF_INITIALIZED = False # Флаг инициализации таблиц смещений

cdef void init_offsets() noexcept nogil:
    """
    Однократное заполнение таблиц смещений лифтинг-шаблонов.
    """
    global OFF_INITIALIZED
    if OFF_INITIALIZED:
        return

    # predict: соседи B-отсчёта, лежащие в косете A
    PRED_OFF[0][0] = 0;  PRED_OFF[0][1] = 0
    PRED_OFF[1][0] = 1;  PRED_OFF[1][1] = -1
    PRED_OFF[2][0] = 0;  PRED_OFF[2][1] = -1
    PRED_OFF[3][0] = 1;  PRED_OFF[3][1] = 0
    PRED_OFF[4][0] = -1; PRED_OFF[4][1] = 0
    PRED_OFF[5][0] = 1;  PRED_OFF[5][1] = -2
    PRED_OFF[6][0] = 0;  PRED_OFF[6][1] = 1
    PRED_OFF[7][0] = 2;  PRED_OFF[7][1] = -1
    PRED_OFF[8][0] = -1; PRED_OFF[8][1] = -1
    PRED_OFF[9][0] = 0;  PRED_OFF[9][1] = -2
    PRED_OFF[10][0] = 1; PRED_OFF[10][1] = 1
    PRED_OFF[11][0] = 2; PRED_OFF[11][1] = 0

    # update: соседи A-отсчёта, лежащие в косете B
    UPD_OFF[0][0] = 0;   UPD_OFF[0][1] = 0
    UPD_OFF[1][0] = -1;  UPD_OFF[1][1] = 1
    UPD_OFF[2][0] = 0;   UPD_OFF[2][1] = 1
    UPD_OFF[3][0] = -1;  UPD_OFF[3][1] = 0
    UPD_OFF[4][0] = -2;  UPD_OFF[4][1] = 1
    UPD_OFF[5][0] = 0;   UPD_OFF[5][1] = -1
    UPD_OFF[6][0] = -1;  UPD_OFF[6][1] = 2
    UPD_OFF[7][0] = 1;   UPD_OFF[7][1] = 0
    UPD_OFF[8][0] = -2;  UPD_OFF[8][1] = 0
    UPD_OFF[9][0] = -1;  UPD_OFF[9][1] = -1
    UPD_OFF[10][0] = 0;  UPD_OFF[10][1] = 2
    UPD_OFF[11][0] = 1;  UPD_OFF[11][1] = 1

    OFF_INITIALIZED = True


cdef inline int wrap(int v, int n) noexcept nogil:
    """
    Быстрое циклическое приведение индекса одним сложением или вычитанием.

    Корректно работает только при -n <= v < 2n, то есть когда индекс вышел за
    границу не более чем на один период.
    
    Замена этой функции на операцию % замедляет направленный банк фильтров в 1.8 раза,
    а полное разложение - в 1.5 раза.

    Args:
        v (int): индекс, возможно вышедший за границы, но находящийся в рамках -n <= v < 2n
        n (int): период приведения (размер массива по данной оси)

    Returns:
        (int): индекс в диапазоне [0, n)
    """
    if v < 0:
        return v + n
    if v >= n:
        return v - n
    return v


cdef inline int cmod(int v, int n) noexcept nogil:
    """
    Полное приведение по модулю с гарантированно неотрицательным
    результатом, для произвольного выхода индекса за границы.

    Вследствие использования % в 1.5 раза медленнее wrap.

    Args:
        v (int): индекс с произвольным выходом за границы
        n (int): период приведения (размер массива по данной оси)

    Returns:
        (int): индекс в диапазоне [0, n)
    """
    v = v % n
    if v < 0:
        v += n
    return v


cdef void lift_forward(double[:, :] a, double[:, :] b) noexcept nogil:
    """
    Прямой лифтинг: шаг предсказания, затем шаг обновления.
    
    Превращает пару сырых квинканс-косетов в пару направленных каналов:
        predict: B <- B - [ A_PRED * (4 ближних A) + B_PRED * (8 дальних A) ]
        update : A <- A + [ A_UPD  * (4 ближних B) + B_UPD  * (8 дальних B) ]

    После шага предсказания канал B содержит ошибку предсказания, то есть
    становится направленным высокочастотным каналом: на гладких участках
    он близок к нулю, а отклик даёт только структура, ориентированная
    поперёк текущего направления. Шаг обновления подмешивает эти остатки
    обратно в A, превращая сырую подвыборку в сглаженный направленный
    низкочастотный канал.

    Args:
        a (2d-memoryview float64, C-совместимый): косет A (p + q чётно);
            на входе - сырая подвыборка, на выходе - направленный низкочастотный канал
        b (2d-memoryview float64, C-совместимый): косет B (p + q нечётно);
            на входе - сырая подвыборка, на выходе - направленный высокочастотный канал
    """
    cdef int M = a.shape[0]
    cdef int Nh = a.shape[1]
    cdef int p, q, t, pp, qq
    cdef double s_near, s_far # Ближние и дальние соседи

    # Predict: косет B становится направленным высокочастотным каналом
    for p in range(M):
        for q in range(Nh):
            s_near = 0.0
            s_far = 0.0
            for t in range(4): # Ближние 4 соседа
                # Смещения в таблице лежат в пределах от −2 до 2,
                # поэтому wrap хватает, cmod не нужен
                pp = wrap(p + PRED_OFF[t][0], M)
                qq = wrap(q + PRED_OFF[t][1], Nh)
                s_near += a[pp, qq]
            for t in range(4, 12): # Дальние 8 соседей
                pp = wrap(p + PRED_OFF[t][0], M)
                qq = wrap(q + PRED_OFF[t][1], Nh)
                s_far += a[pp, qq]
            # Предсказание и вычитание - остаётся только ошибка угадывания
            b[p, q] -= A_PRED * s_near + B_PRED * s_far

    # Update: косет A становится направленным низкочастотным каналом
    for p in range(M):
        for q in range(Nh):
            s_near = 0.0
            s_far = 0.0
            for t in range(4):
                pp = wrap(p + UPD_OFF[t][0], M)
                qq = wrap(q + UPD_OFF[t][1], Nh)
                s_near += b[pp, qq]
            for t in range(4, 12):
                pp = wrap(p + UPD_OFF[t][0], M)
                qq = wrap(q + UPD_OFF[t][1], Nh)
                s_far += b[pp, qq]
            # Заполнение косета A, который становится корректно сглаженным низкочастотным каналом
            a[p, q] += A_UPD * s_near + B_UPD * s_far


cdef void lift_inverse(double[:, :] a, double[:, :] b) noexcept nogil:
    """
    Обратный лифтинг: обратный шаг обновления, затем обратный шаг предсказания.

    Превращает пару направленных каналов обратно в пару сырых
    квинканс-косетов:
        обратный update : A <- A - [ A_UPD  * (4 ближних B) + B_UPD  * (8 дальних B) ]
        обратный predict: B <- B + [ A_PRED * (4 ближних A) + B_PRED * (8 дальних A) ]

    Строго обратна lift_forward(): те же две таблицы смещений и те же
    четыре веса, но шаги идут в противоположном порядке и с обратными
    знаками. Композиция даёт тождество при любых значениях весов.

    Args:
        a (2d-memoryview float64, C-совместимый):
            на входе - направленный низкочастотный канал,
            на выходе - сырой косет A (p + q чётно).
        b (2d-memoryview float64, C-совместимый):
            на входе - направленный высокочастотный канал,
            на выходе - сырой косет B (p + q нечётно).
    """
    cdef int M = a.shape[0]
    cdef int Nh = a.shape[1]
    cdef int p, q, t, pp, qq
    cdef double s_near, s_far # Ближние и дальние соседи

    # Обратный Update
    for p in range(M):
        for q in range(Nh):
            s_near = 0.0
            s_far = 0.0
            for t in range(4):
                pp = wrap(p + UPD_OFF[t][0], M)
                qq = wrap(q + UPD_OFF[t][1], Nh)
                s_near += b[pp, qq]
            for t in range(4, 12):
                pp = wrap(p + UPD_OFF[t][0], M)
                qq = wrap(q + UPD_OFF[t][1], Nh)
                s_far += b[pp, qq]
            # Вычитаем - в прямом Update прибавляли
            a[p, q] -= A_UPD * s_near + B_UPD * s_far

    # Оратный Predict
    for p in range(M):
        for q in range(Nh):
            s_near = 0.0
            s_far = 0.0
            for t in range(4):
                pp = wrap(p + PRED_OFF[t][0], M)
                qq = wrap(q + PRED_OFF[t][1], Nh)
                s_near += a[pp, qq]
            for t in range(4, 12):
                pp = wrap(p + PRED_OFF[t][0], M)
                qq = wrap(q + PRED_OFF[t][1], Nh)
                s_far += a[pp, qq]
            # Прибавляем - в прямом Predict вычитали
            b[p, q] += A_PRED * s_near + B_PRED * s_far


cdef void qx_split(double[:, :] x, double[:, :] a, double[:, :] b,
                    int modulate) noexcept nogil:
    """
    Анализ одной ступени квинканс-веерного банка фильтров: разделение
    входного массива на два направленных поддиапазона вдвое меньшей ширины.

    2 этапа:
    1) полифазное расщепление по квинканс-косетам с одновременной
        модуляцией, переводящей ромбовидный фильтр в веерный.
    2) прямой лифтинг (lift_forward), превращающий сырые косеты
           в направленные низко- и высокочастоные каналы.

    Args:
        x (2d-memoryview float64, C-совместимый): входной массив;
            N чётное, N делит M
        a (2d-memoryview float64, C-совместимый): выходной направленный низкочастотный канал
        b (2d-memoryview float64, C-совместимый): выходной направленный высокочастотный канал
        modulate (int):
            1 - применить модуляцию (-1)^i, дающую веерные (направленные) фильтры.
            0 - без модуляции, ромбовидные (чисто частотные) фильтры.
    """
    cdef int M = x.shape[0]
    cdef int N = x.shape[1]
    cdef int Nh = N // 2
    # i, j0, j1 - координаты в исходной сетке
    # p, q - в решёточной
    cdef int p, q, i, j0, j1
    cdef double sgn

    # Полифазное расщепление по квинканс-косетам в решёточных координатах
    for p in range(M):
        for q in range(Nh):
            # Номер строки в исходном массиве
            i = wrap(p + q, M) # Сумма p + q не достигает 2M, поэтому wrap хватит
            # Номер столбца в исходном массиве
            j0 = cmod(p - q, N)
            # Соседний справа столбец
            j1 = j0 + 1 if j0 + 1 < N else 0
            # Модуляция (-1)^i задаётся исходным номером строки i = (p+q) mod M
            sgn = -1.0 if (modulate != 0 and ((p + q) & 1) == 1) else 1.0
            # Применение модуляции
            a[p, q] = sgn * x[i, j0]
            b[p, q] = sgn * x[i, j1]

    lift_forward(a, b) # Прямой лифтинг: сырые косеты в направленные каналы


cdef void qx_merge(double[:, :] a, double[:, :] b, double[:, :] x,
                    int modulate) noexcept nogil:
    """
    Синтез одной ступени квинканс-веерного банка фильтров: сборка двух
    направленных поддиапазонов обратно в массив полной ширины.

    2 этапа:
        1) обратный лифтинг, возвращающий направленные
           каналы в состояние сырых квинканс-косетов.
        2) обратное полифазное объединение с одновременным снятием
           модуляции.

    Args:
        a (2d-memoryview float64, C-совместимый): направленный низкочастотный канал
        b (2d-memoryview float64, C-совместимый): направленный высокочастотный канал
        x (2d-memoryview float64, C-совместимый): выходной массив полной ширины
        modulate (int): снятие модуляции.
            Значение должно совпадать с использованным в qx_split.
    """
    cdef int M = x.shape[0]
    cdef int N = x.shape[1]
    cdef int Nh = N // 2
    # i, j0, j1 - координаты в исходной сетке
    # p, q - в решёточной
    cdef int p, q, i, j0, j1
    cdef double sgn

    lift_inverse(a, b) # Обратный лифтинг (в qx_split лифтинг был 2-ым шагом, здесь 1-ым)

    # Обратное полифазное объединение
    for p in range(M):
        for q in range(Nh):
            i = wrap(p + q, M)
            j0 = cmod(p - q, N)
            j1 = j0 + 1 if j0 + 1 < N else 0
            sgn = -1.0 if (modulate != 0 and ((p + q) & 1) == 1) else 1.0
            # Снятие модуляции
            x[i, j0] = sgn * a[p, q]
            x[i, j1] = sgn * b[p, q]


def dfb_analysis(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] band, int levels):
    """
    Python-функция, т.к. число поддиапазонов зависит от levels,
    которое определяется в main.
    
    Направленное разложение полосового остатка на 2^levels клиновидных
    поддиапазонов - двоичное дерево квинканс-веерных банков фильтров.

    Рекурсивно применяет qx_split(): каждый уровень расщепляет массив
    надвое по ширине, а спектральный клин каждого канала поворачивается
    на 45 градусов относительно родительского.

    Рекомендуемое значение levels - 1 или 2. При levels >= 3 корректное
    выравнивание клиньев требует матриц пересдискретизации,
    которые здесь не реализованы. 8 каналов будут получены, восстановление
    останется точным, но хорошо разделены окажутся лишь около четырёх из них.

    Args:
        band (2d-array float64, C-совместимый): полосовой остаток
            Лапласовой пирамиды или произвольный массив.
            N чётное, N делит M
        levels (int): число уровней дерева.
            При levels <= 0 разложение не выполняется.

    Returns:
        (list): список из 2^levels массивов float64, C-совместимых, формы.
        Порядок соответствует обходу дерева слева направо:
            первая половина списка - потомки низкочастотного канала,
            вторая - потомки высокочастотного.
    """
    if levels <= 0:
        return [band] # Сразу возвращаем, если раскладывать не нужно

    cdef int M = band.shape[0]
    cdef int N = band.shape[1]

    # Проверки на корректность размеров полосового остатка
    if N % 2 != 0:
        raise ValueError(f"Ширина {N} не делится на 2: невозможно продолжить DFB.")
    if M % N != 0:
        raise ValueError(
            f"Форма {M}x{N}: требуется, чтобы N делило M (условие обратимости "
            f"решёточного представления квинканс-косета). Используйте квадратное "
            f"изображение со стороной, кратной 2^(n_levels + dfb_levels).")

    init_offsets() # Инициализация таблиц смещений

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] a = np.empty((M, N // 2), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] b = np.empty((M, N // 2), dtype=np.float64)
    # Работа с memoryview
    cdef double[:, :] v_x = band
    cdef double[:, :] v_a = a
    cdef double[:, :] v_b = b

    with nogil:
        qx_split(v_x, v_a, v_b, 1) # Собственно разложение на 2 клиновидных поддиапазона

    # Рекурсивный вызов для продления на n уровней
    return dfb_analysis(a, levels - 1) + dfb_analysis(b, levels - 1)


def dfb_synthesis(list subbands, int levels):
    """
    Обратное направленное разложение: сборка 2^levels клиновидных
    поддиапазонов обратно в полосовой остаток полной ширины.

    Рекурсивно применяет qx_merge(), проходя двоичное дерево от листьев
    к корню. На каждом уровне список разрезается пополам, половины
    собираются рекурсивно, и два полученных массива объединяются
    в один вдвое большей ширины.

    Критически зависит от порядка элементов списка:
    первая половина должна содержать потомков низкочастотного канала,
    вторая - высокочастотного.

    Args:
        subbands (list): список из 2^levels массивов float64;
            допустимы любые объекты, приводимые np.ascontiguousarray к нужному типу.
        levels (int): число уровней дерева.
            Должно совпадать со значением, использованным в dfb_analysis.
            При levels <= 0 возвращается первый элемент списка.

    Returns:
        out (2d-array float64, C-совместимый): восстановленный полосовой остаток.
    """
    if levels <= 0:
        return np.ascontiguousarray(subbands[0], dtype=np.float64) # Возврат первого элемента

    init_offsets() # Инициализация таблиц смещений

    cdef int half = len(subbands) // 2 # Разрез списка посередине
    # Рекурсивыне вызовы на левую и правую половины
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] a = np.ascontiguousarray(
        dfb_synthesis(subbands[:half], levels - 1), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] b = np.ascontiguousarray(
        dfb_synthesis(subbands[half:], levels - 1), dtype=np.float64)

    cdef int M = a.shape[0]
    cdef int N = a.shape[1]
    # Буфер для выхода
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] out = np.empty((M, 2 * N), dtype=np.float64)

    # Работа с memoryview
    cdef double[:, :] v_a = a
    cdef double[:, :] v_b = b
    cdef double[:, :] v_out = out

    with nogil:
        qx_merge(v_a, v_b, v_out, 1) # Непосредственно сборка

    return out


# ============================================================================
#  Раздел 3: Полное контурлет-преобразование
# ============================================================================

def contourlet_decompose(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] image,
                         int n_levels=2, int dfb_levels=2):
    """
    Python-функция, т.к. возвращается кортеж и выбрасываются исключения.

    Прямое контурлет-преобразование = лапласова пирамида + DFB на каждом
    полосовом уровне.

    Схема работы - каскад из двух независимых слоёв. Пирамида разделяет
    изображение по масштабам: на каждом шаге выделяется грубое приближение
    вдвое меньшего размера и полосовой остаток полного разрешения.
    Направленный банк фильтров разделяет каждый такой остаток
    по направлениям, на 2^dfb_levels клиновидных поддиапазона.

    Args:
        image (2d-array float64, C-совместимый): входное изображение.
        n_levels (int): число уровней лапласовой пирамиды; должно быть >= 1.
        dfb_levels (int): число уровней направленного дерева; рекомендуется 1 или 2.

    Returns:
        (lowpass, bands) - кортеж:
            lowpass (2d-array float64, C-совместимый): грубое приближение,
                остаток пирамиды после всех уровней.
            bands (list): список списков, каждый из которых содержит
                2^dfb_levels направленных поддиапазонов масштаба k,
                где k = 0 соответствует самому мелкому масштабу,
                а k = n_levels - 1 - самому грубому.
    """
    init_filters() # Инициализация коэффициентов фильтра

    cdef int H = image.shape[0]
    cdef int W = image.shape[1]
    cdef int k
    cdef int need = 1 << n_levels # Минимальная кратность стороны входного массива

    if n_levels < 1:
        raise ValueError("n_levels должно быть >= 1.")
    if H % need != 0 or W % need != 0:
        raise ValueError(
            f"Размер {W}x{H} не кратен 2^n_levels = {need}. Обрежьте изображение.")

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] cur = image # Текущее изображеие
    bands = [] # Выход функции

    for k in range(n_levels):
        # Один шаг Лапласовой пирамиды
        coarse, band = lp_analysis_step(cur)
        # Направленное разложение полосового остатка данного масштаба
        bands.append(dfb_analysis(band, dfb_levels))
        # Переход к следующему (более грубому) масштабу
        cur = coarse

    return cur, bands


def contourlet_reconstruct(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] lowpass,
                           list bands, int dfb_levels=2):
    """
    Обратное контурлет-преобразование: сборка изображения из грубого
    приближения и направленных поддиапазонов всех масштабов.

    Обход идёт в противоположном порядке - от самого грубого масштаба к самому мелкому.
    На каждом шаге направленные поддиапазоны собираются обратно в полосовой
    остаток, после чего остаток складывается с приближением, приходящим
    с более грубого уровня.

    Args:
        lowpass (2d-array float64, C-совместимый): грубое приближение;
            первый элемент кортежа из contourlet_decompose.
        bands (list): список списков, каждый из которых содержит
            2^dfb_levels направленных поддиапазонов масштаба k.
            Порядок элементов внутри каждого списка обязан совпадать
            с тем, что вернул dfb_analysis.
        dfb_levels (int): число уровней направленного дерева.
            Должно совпадать со значением, использованным в contourlet_decompose.

    Returns:
        cur (2d-array float64, C-совместимый): восстановленное изображение
    """
    init_filters() # Инициализация коэффициентов фильтра

    cdef int k
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] cur = lowpass # Текущее изображеие
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] band

    # От грубого масштаба к мелкому
    for k in range(len(bands) - 1, -1, -1):
        # Yаправленные поддиапазоны данного масштаба собираются в полосовой остаток
        band = np.ascontiguousarray(dfb_synthesis(bands[k], dfb_levels),
                                    dtype=np.float64)
        # Один уровень синтеза пирамиды
        cur = np.ascontiguousarray(lp_synthesis_step(cur, band), dtype=np.float64)

    return cur


# ============================================================================
#  Раздел 4: Встраивание и извлечение ЦВЗ
# ============================================================================

cdef inline int capacity(int h, int w, int block, int n_sub) noexcept nogil:
    """
    Ёмкость масштаба: число блоков во всех его направленных поддиапазонах.

    Args:
        h (int): высота направленного поддиапазона
        w (int): ширина направленного поддиапазона
        block (int): сторона блока, >= 4
        n_sub (int): число направленных поддиапазонов масштаба,
            равно 2^dfb_levels

    Returns:
        (int): максимальное число бит, которое можно встроить в данный
            масштаб
    """
    return n_sub * (h // block) * (w // block)


cdef void embed_subband(double[:, :] S, cnp.int32_t[:] wm, int wm_len,
                         int sub_i, int n_sub, double margin,
                         int block) noexcept nogil:
    """
    Встраивание бит в один направленный поддиапазон.

    Поддиапазон разбивается на непересекающиеся блоки, в каждом берётся
    пара коэффициентов на фиксированных позициях. Бит кодируется знаком их разности
    (лучшая устойчивость к общим атакам на изображение) с зазором margin.

    Обрабатываются только биты, закреплённые за этим поддиапазоном
    циклическим распределением: i = blk * n_sub + sub_i.

    Args:
        S (2d-memoryview float64, C-совместимый): направленный поддиапазон.
        wm (1d-memoryview int32): биты ЦВЗ.
        wm_len (int): длина ЦВЗ.
        sub_i (int): номер данного поддиапазона, 0 <= sub_i < n_sub.
        n_sub (int): общее число поддиапазонов масштаба.
        margin (double): требуемый зазор между парой коэффициентов;
            чем больше тем выше устойчивость, но ниже PSNR.
        block (int): сторона блока, >= 4.
    """
    cdef int h = S.shape[0]
    cdef int w = S.shape[1]
    cdef int nb_c = w // block # Число блоков в строке
    cdef int n_blocks = (h // block) * nb_c # Всего блоков в поддиапазоне
    cdef int blk, i_bit, br, bc, r0, k0
    cdef double c1, c2, d

    for blk in range(n_blocks):
        i_bit = blk * n_sub + sub_i # В какие биты встраиваем
        if i_bit >= wm_len:
            return # Когда встроили весь ЦВЗ - выходим

        # Пееревод сквозного номера в координаты левого верхнего угла блока
        br = blk // nb_c
        bc = blk % nb_c
        r0 = br * block
        k0 = bc * block

        # Коэффициенты для встраивания
        c1 = S[r0 + 1, k0 + 1]
        c2 = S[r0 + 2, k0 + 2]

        # Встраивание бита ЦВЗ
        if wm[i_bit] == 1:
            # Требуется c1 - c2 >= margin, поэтому ряд блоков можно не трогать
            if c1 - c2 < margin:
                d = 0.5 * (margin - (c1 - c2))
                S[r0 + 1, k0 + 1] = c1 + d
                S[r0 + 2, k0 + 2] = c2 - d
        else:
            # Требуется c2 - c1 >= margin
            if c2 - c1 < margin:
                d = 0.5 * (margin - (c2 - c1))
                S[r0 + 2, k0 + 2] = c2 + d
                S[r0 + 1, k0 + 1] = c1 - d


cdef void extract_subband(double[:, :] S, cnp.int32_t[:] wm, int wm_len,
                           int sub_i, int n_sub, int block) noexcept nogil:
    """
    Извлечение бит из одного направленного поддиапазона.

    Args:
        S (2d-memoryview float64, C-совместимый): направленный поддиапазон изображения с ЦВЗ.
        wm (1d-memoryview int32): выходной массив бит ЦВЗ;
            заполняются только позиции, закреплённые за данным поддиапазоном.
        wm_len (int): длина ЦВЗ.
        sub_i (int): номер данного поддиапазона, 0 <= sub_i < n_sub.
        n_sub (int): общее число поддиапазонов масштаба.
        block (int): сторона блока;
            обязана совпадать со значением, использованным в embed_subband.
    """
    cdef int h = S.shape[0]
    cdef int w = S.shape[1]
    cdef int nb_c = w // block # Число блоков в строке
    cdef int n_blocks = (h // block) * nb_c # Всего блоков в поддиапазоне
    cdef int blk, i_bit, br, bc, r0, k0
    cdef double c1, c2

    for blk in range(n_blocks):
        i_bit = blk * n_sub + sub_i # Из каких бит извлекаем
        if i_bit >= wm_len:
            return # Когда извлекли весь ЦВЗ - выходим

        # Пееревод сквозного номера в координаты левого верхнего угла блока
        br = blk // nb_c
        bc = blk % nb_c
        r0 = br * block
        k0 = bc * block

        # Коэффициенты для встраивания
        c1 = S[r0 + 1, k0 + 1]
        c2 = S[r0 + 2, k0 + 2]

        wm[i_bit] = 1 if c1 > c2 else 0 # Определение бита ЦВЗ


def embed_watermark_contourlet(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] image,
                               cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] watermark,
                               double margin=24.0,
                               int n_levels=2,
                               int dfb_levels=2,
                               int scale=-1,
                               int block=4,
                               int iterations=3):
    """
    Встраивание ЦВЗ в контурлет-область.

    Полный цикл на каждой итерации: прямое контурлет-преобразование,
    модификация коэффициентов выбранного масштаба, обратное преобразование.

    Философия итерирования служит для улучшения BER и PSNR.

    Args:
        image (2d-array float64, C-совместимый: чистое изображение.
        watermark (1d-array int32, C-совместимый): биты ЦВЗ.
        margin (double): требуемый зазор между парой коэффициентов;
            чем больше тем выше устойчивость, но ниже PSNR.
        n_levels (int): число масштабов Лапласовой пирамиды.
        dfb_levels (int): число уровней направленного дерева.
        scale (int): индекс масштаба для встраивания;
            -1 означает самый грубый полосовой уровень,
            наиболее устойчивый к сжатию и сглаживанию.
        block (int): сторона блока внутри поддиапазона, >= 4.
        iterations (int): число итераций встраивания, >= 1.

    Returns:
        cur (2d-array float64, C-совместимый): изображение с внедрённым ЦВЗ.
    """
    init_filters() # Инициализация коэффициентов фильтра

    if block < 4:
        raise ValueError("block должен быть >= 4.")
    if iterations < 1:
        raise ValueError("iterations должно быть >= 1.")

    cdef int wm_len = watermark.shape[0]
    # Разворачивание отрицательного scale в конкретный индекс
    cdef int sc = n_levels - 1 if scale < 0 else scale
    if sc < 0 or sc >= n_levels:
        raise ValueError(f"scale={sc} вне диапазона [0, {n_levels - 1}].")

    cdef cnp.int32_t[:] wm_view = watermark # Работа с memoryview
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] cur = image # Текущее изображение
    cdef double[:, :] S
    cdef int s_i, n_sub, h, w, cap, it

    for it in range(iterations):
        # Прямое контурлет-преобразование
        lowpass, bands = contourlet_decompose(cur, n_levels, dfb_levels)
        subbands = bands[sc] # Выбор масштаба
        n_sub = len(subbands)
        h = subbands[0].shape[0]
        w = subbands[0].shape[1]

        if it == 0:
            # Проверка на вмещаемость ЦВЗ
            cap = capacity(h, w, block, n_sub)
            if wm_len > cap:
                raise ValueError(
                    f"Задано {wm_len} бит, но ёмкость масштаба {sc} только {cap} бит "
                    f"({n_sub} поддиапазонов {h}x{w}, блок {block}x{block}).")

        # Цикл по направленным поддиапазонам
        for s_i in range(n_sub):
            S = subbands[s_i]
            with nogil:
                embed_subband(S, wm_view, wm_len, s_i, n_sub, margin, block) # Встраивание

        # Обратное контурлет-преобразование
        cur = np.ascontiguousarray(
            contourlet_reconstruct(np.ascontiguousarray(lowpass, dtype=np.float64),
                                   bands, dfb_levels),
            dtype=np.float64)

    return cur


def extract_watermark_contourlet(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] image,
                                 int wm_length,
                                 int n_levels=2,
                                 int dfb_levels=2,
                                 int scale=-1,
                                 int block=4):
    """
    Извлечение ЦВЗ из контурлет-области.

    Args:
        image (2d-array float64, C-совместимый): изображение, содержащее ЦВЗ.
        wm_length (int): длина ЦВЗ в битах.
        n_levels (int): число масштабов лапласовой пирамиды.
            Должно совпадать со значением в embed_watermark_contourlet.
        dfb_levels (int): число уровней направленного дерева.
            Должно совпадать со значением в embed_watermark_contourlet.
        scale (int): индекс масштаба.
            Должно совпадать со значением в embed_watermark_contourlet.
        block (int): сторона блока внутри поддиапазона.
            Должно совпадать со значением в embed_watermark_contourlet.

    Returns:
        extracted_wm (1d-array int32, C-совместимый): извлечённые биты ЦВЗ.
    """
    init_filters() # Инициализация коэффициентов фильтра

    if block < 4:
        raise ValueError("block должен быть >= 4.")

    # Разворачивание отрицательного scale в конкретный индекс
    cdef int sc = n_levels - 1 if scale < 0 else scale
    if sc < 0 or sc >= n_levels:
        raise ValueError(f"scale={sc} вне диапазона [0, {n_levels - 1}].")

    # Прямое контурлет-преобразование и выбор масштаба
    lowpass, bands = contourlet_decompose(image, n_levels, dfb_levels)
    subbands = bands[sc]

    cdef int n_sub = len(subbands)
    cdef int h = subbands[0].shape[0]
    cdef int w = subbands[0].shape[1]
    cdef int cap = capacity(h, w, block, n_sub)

    # Проверка на заданную длину ЦВЗ
    if wm_length > cap:
        raise ValueError(
            f"Задано {wm_length} бит, но ёмкость масштаба {sc} только {cap} бит.")

    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] extracted_wm = np.zeros(
        wm_length, dtype=np.int32)
    cdef cnp.int32_t[:] wm_view = extracted_wm # Работа с memoryview

    cdef double[:, :] S
    cdef int s_i
    # Цикл по направленным поддиапазонам
    for s_i in range(n_sub):
        S = subbands[s_i]
        with nogil:
            extract_subband(S, wm_view, wm_length, s_i, n_sub, block) # Извлечение

    return extracted_wm

class Contourlet(Ready_Frequency_Embeddings):
    """
    Встраивание ЦВЗ в контурлет-область.

    Лапласова пирамида даёт многомасштабное разложение, направленная блок
    фильтров (DFB) режет каждый полосовой уровень на направленные
    поддиапазоны. Бит кодируется зазором между парой коэффициентов внутри
    блока поддиапазона. В отличие от вейвлетов, базис анизотропен и следует
    контурам изображения, поэтому правки лучше маскируются на текстурах.

    Встраивание итеративное: после обратного преобразования коэффициенты
    смещаются, и цикл повторяется, что заметно улучшает BER при том же PSNR.
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
                margin (float): требуемый зазор между парой коэффициентов (по умолчанию 24.0)
                n_levels (int): число масштабов лапласовой пирамиды (по умолчанию 2)
                dfb_levels (int): число уровней направленного дерева (по умолчанию 2)
                scale (int): индекс масштаба; -1 означает самый грубый полосовой уровень (по умолчанию -1)
                block (int): сторона блока внутри поддиапазона, не меньше 4 (по умолчанию 4)
                iterations (int): число итераций встраивания, не меньше 1 (по умолчанию 3)

        Returns:
            None
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        watermark = bits_to_array(args["watermark_bits"])
        margin = float(args.get("margin", 24.0))
        n_levels = int(args.get("n_levels", 2))
        dfb_levels = int(args.get("dfb_levels", 2))
        scale = int(args.get("scale", -1))
        block = int(args.get("block", 4))
        iterations = int(args.get("iterations", 3))

        luma, chroma = load_luma(input_data)
        save_luma(
            embed_watermark_contourlet(luma, watermark, margin, n_levels,
                                       dfb_levels, scale, block, iterations),
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

        Параметры разложения обязаны совпадать со значениями, использованными
        при встраивании: детектор строит то же дерево поддиапазонов.

        Args:
            args (dict): параметры извлечения
                input_data (str): путь к изображению с ЦВЗ
                num_bits (int): длина извлекаемого ЦВЗ в битах
                n_levels (int): число масштабов лапласовой пирамиды (по умолчанию 2)
                dfb_levels (int): число уровней направленного дерева (по умолчанию 2)
                scale (int): индекс масштаба (по умолчанию -1)
                block (int): сторона блока внутри поддиапазона (по умолчанию 4)

        Returns:
            str: строка из '0' и '1' длины num_bits
        """
        input_data = args["input_data"]
        num_bits = int(args["num_bits"])
        n_levels = int(args.get("n_levels", 2))
        dfb_levels = int(args.get("dfb_levels", 2))
        scale = int(args.get("scale", -1))
        block = int(args.get("block", 4))

        luma, _ = load_luma(input_data)
        return array_to_bits(
            extract_watermark_contourlet(luma, num_bits, n_levels,
                                         dfb_levels, scale, block)
        )
