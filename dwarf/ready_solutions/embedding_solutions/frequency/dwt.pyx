import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy
from ...utils.embedding_utils import *

cnp.import_array()

cdef void get_wavelet_filters(char* name, double* h, double* g, int* L):
    """
    Инициализирует фильтры вейвлетов.

    Args:
        name (char*): имя вейвлета
        h (double*): указатель на массив низкочастотных коэффициентов
        g (double*): указатель на массив высокочастотных коэффициентов
        L (int*): указатель на длину фильтра
    """
    if name == b"haar":
        L[0] = 2
        h[0] =  0.7071067811865476; h[1] =  0.7071067811865476
        g[0] =  0.7071067811865476; g[1] = -0.7071067811865476
    elif name == b"db4":
        L[0] = 8
        h[0] =  0.2303778133088964; h[1] =  0.7148465705529154
        h[2] =  0.6308807679398587; h[3] = -0.0279837694168599
        h[4] = -0.1870348117190931; h[5] =  0.0308413818355607
        h[6] =  0.0328830116668852; h[7] = -0.0105974017850690
        g[0] = -0.0105974017850690; g[1] = -0.0328830116668852
        g[2] =  0.0308413818355607; g[3] =  0.1870348117190931
        g[4] = -0.0279837694168599; g[5] = -0.6308807679398587
        g[6] =  0.7148465705529154; g[7] = -0.2303778133088964
    elif name == b"sym4":
        L[0] = 8
        h[0] = -0.0757657147893432; h[1] = -0.0296355276459985
        h[2] =  0.4976186676323012; h[3] =  0.8037387518059161
        h[4] =  0.2978577956052774; h[5] = -0.0992195435768472
        h[6] = -0.0126039672622612; h[7] =  0.0322231006040427
        g[0] =  0.0322231006040427; g[1] =  0.0126039672622612
        g[2] = -0.0992195435768472; g[3] = -0.2978577956052774
        g[4] =  0.8037387518059161; g[5] = -0.4976186676323012
        g[6] = -0.0296355276459985; g[7] =  0.0757657147893432
    else:
        raise ValueError(f"Неизвестный вейвлет: {name}")

cdef inline void dwt_1d_haar(const double* x, double* a, double* d, int n):
    """
    Прямое 1D DWT-преобразование для Haar (без цикла свёртки для ускорения).

    Args:
        x (const double*): входной сигнал
        a (double*): выходной массив аппроксимации
        d (double*): выходной массив деталей
        n (int): длина входного сигнала
    """
    cdef int i, idx0, idx1
    cdef double x0, x1
    cdef double inv_sqrt2 = 0.7071067811865476
    for i in range(n >> 1): # То же, что и n//2
        idx0 = i << 1
        idx1 = idx0 + 1
        if idx1 >= n: idx1 -= n
        x0 = x[idx0]
        x1 = x[idx1]
        a[i] = (x0 + x1) * inv_sqrt2 # Коэффициент аппроксимации
        d[i] = (x0 - x1) * inv_sqrt2 # Коэффициент деталей

cdef inline void idwt_1d_haar(const double* a, const double* d, double* x, int n):
    """
    Обратное 1D DWT-преобразование для Haar (без цикла свёртки для ускорения).

    Args:
        a (const double*): массив аппроксимации
        d (const double*): массив деталей
        x (double*): выходной восстановленный сигнал
        n (int): длина выходного сигнала
    """
    cdef int i, idx0, idx1
    cdef double a_val, d_val
    cdef double inv_sqrt2 = 0.7071067811865476
    for i in range(n >> 1):
        a_val = a[i] * inv_sqrt2
        d_val = d[i] * inv_sqrt2
        idx0 = i << 1
        idx1 = idx0 + 1
        if idx1 >= n: idx1 -= n
        x[idx0] = a_val + d_val # Результат реставрации чётных отсчётов
        x[idx1] = a_val - d_val # Результат реставрации нечётных отсчётов

cdef inline void dwt_1d_l8(const double* x, double* a, double* d, int n, const double* h, const double* g):
    """
    Прямое 1D DWT-преобразование для Daubechies и Symlets.

    Args:
        x (const double*): входной сигнал
        a (double*): выходной массив аппроксимации
        d (double*): выходной массив деталей
        n (int): длина входного сигнала
        h (const double*): низкочастотный фильтр
        g (const double*): высокочастотный фильтр
    """
    cdef int i, k, idx
    for i in range(n >> 1):
        a[i] = 0.0
        d[i] = 0.0
        for k in range(8):
            idx = (i << 1) + k
            if idx >= n: idx -= n
            a[i] += h[k] * x[idx] # Коэффициент аппроксимации
            d[i] += g[k] * x[idx] # Коэффициент деталей

cdef inline void idwt_1d_l8(const double* a, const double* d, double* x, int n, const double* h, const double* g):
    """
    Обратное 1D DWT-преобразование для Daubechies и Symlets.

    Args:
        a (const double*): массив аппроксимации
        d (const double*): массив деталей
        x (double*): выходной восстановленный сигнал
        n (int): длина выходного сигнала
        h (const double*): низкочастотный фильтр
        g (const double*): высокочастотный фильтр
    """
    cdef int i, k, m
    cdef int half_n = n >> 1
    for i in range(n):
        x[i] = 0.0
        for k in range(half_n):
            m = i - (k << 1)
            if m < 0: m += n
            if m < 8:
                x[i] += h[m] * a[k] + g[m] * d[k] # Результат алгоритма Малла

cdef void dwt_2d_image(double[:, :] img,
                       double[:, :] LL, double[:, :] LH, double[:, :] HL, double[:, :] HH, 
                       double* temp, double* row_a, double* row_d,
                       double* col_in, double* col_a, double* col_d,
                       const double* h, const double* g, int L):
    """
    Прямое 2D DWT-преобразование для всего изображения.
    
    Args:
        img (2d-array float64, C-совместимый): входное изображение
        LL, LH, HL, HH (2d-array float64, C-совместимые): выходные подполосы
        temp (1d-array float64): временный буфер
        row_a, row_d (1d-array float64): буферы аппроксимации и деталей строк
        col_in, col_a, col_d (1d-array float64): буферы входного столбца, аппроксимации и деталей столбцов
        h, g (const double*): низкочастотный и высокочастотный фильтры
        L (int): длина фильтра
    """
    cdef int H = img.shape[0]
    cdef int W = img.shape[1]
    cdef int half_H = H >> 1
    cdef int half_W = W >> 1
    cdef int r, c, idx
    
    if L == 2: # Haar
        for r in range(H): # DWT по строкам
            dwt_1d_haar(&img[r, 0], row_a, row_d, W)
            idx = r * W
            for c in range(half_W): # Запись в temp аппроксимации и деталей
                temp[idx + c] = row_a[c]
                temp[idx + half_W + c] = row_d[c]
        for c in range(W): # DWT по столбцам
            for r in range(H):
                col_in[r] = temp[r * W + c] # Текущий столбец
            dwt_1d_haar(col_in, col_a, col_d, H)
            for r in range(half_H):
                if c < half_W: # Разделение на записи в соответствующие подполосы
                    LL[r, c] = col_a[r]
                    HL[r, c] = col_d[r]
                else:
                    LH[r, c - half_W] = col_a[r]
                    HH[r, c - half_W] = col_d[r]
    else:
        # Daubechies и Symlets
        for r in range(H):
            dwt_1d_l8(&img[r, 0], row_a, row_d, W, h, g)
            idx = r * W
            for c in range(half_W):
                temp[idx + c] = row_a[c]
                temp[idx + half_W + c] = row_d[c]
        for c in range(W):
            for r in range(H):
                col_in[r] = temp[r * W + c]
            dwt_1d_l8(col_in, col_a, col_d, H, h, g)
            for r in range(half_H):
                if c < half_W:
                    LL[r, c] = col_a[r]
                    HL[r, c] = col_d[r]
                else:
                    LH[r, c - half_W] = col_a[r]
                    HH[r, c - half_W] = col_d[r]

cdef void idwt_2d_image(double[:, :] LL, double[:, :] LH, double[:, :] HL, double[:, :] HH,
                        double[:, :] img, 
                        double* temp, double* col_a, double* col_d, double* col_out, 
                        double* row_a, double* row_d, double* row_out,
                        const double* h, const double* g, int L):
    """
    Обратное 2D DWT-преобразование для всего изображения.
    
    Args:
        LL, LH, HL, HH (2d-array float64, C-совместимые): входные подполосы
        img (2d-array float64, C-совместимый): выходное реставрированное изображение
        temp (1d-array float64): временный буфер
        col_a, col_d, col_out (1d-array float64): буферы аппроксимации и деталей столбцов и выходного столбца
        row_a, row_d, row_out (1d-array float64): буферы аппроксимации и деталей строк и выходной строки
        h, g (const double*): низкочастотный и высокочастотный фильтры
        L (int): длина фильтра
    """
    cdef int H = img.shape[0]
    cdef int W = img.shape[1]
    cdef int half_H = H >> 1
    cdef int half_W = W >> 1
    cdef int r, c
    
    if L == 2: # Haar
        for c in range(W): # IDWT по столбцам
            for r in range(half_H):
                if c < half_W: # Запись в буфер столбца аппроксимации и деталей
                    col_a[r] = LL[r, c]
                    col_d[r] = HL[r, c]
                else:
                    col_a[r] = LH[r, c - half_W]
                    col_d[r] = HH[r, c - half_W]
            idwt_1d_haar(col_a, col_d, col_out, H)
            for r in range(H): # Запись столбцов в temp
                temp[r * W + c] = col_out[r]

        for r in range(H): # IDWT по строкам
            for c in range(half_W): # Запись в буфер строки аппроксимации и деталей
                row_a[c] = temp[r * W + c]
                row_d[c] = temp[r * W + half_W + c]
            idwt_1d_haar(row_a, row_d, row_out, W)
            memcpy(&img[r, 0], row_out, W * sizeof(double)) # Запись строк в img
    else: # Daubechies и Symlets
        for c in range(W):
            for r in range(half_H):
                if c < half_W:
                    col_a[r] = LL[r, c]
                    col_d[r] = HL[r, c]
                else:
                    col_a[r] = LH[r, c - half_W]
                    col_d[r] = HH[r, c - half_W]
            idwt_1d_l8(col_a, col_d, col_out, H, h, g)
            for r in range(H):
                temp[r * W + c] = col_out[r]
        for r in range(H):
            for c in range(half_W):
                row_a[c] = temp[r * W + c]
                row_d[c] = temp[r * W + half_W + c]
            idwt_1d_l8(row_a, row_d, row_out, W, h, g)
            memcpy(&img[r, 0], row_out, W * sizeof(double))

def embed_watermark_dwt(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] image, 
                        cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] watermark, 
                        double margin=15.0,
                        str wavelet_name="sym4"):
    """
    Встраивание ЦВЗ.
    
    Args:
        image (2d-array float64, C-совместимый): чистое изображение
        watermark (1D-array int32): ЦВЗ
        margin (double): изменение коэффициентов при непосредственном встраивании
        wavelet_name (str): тип вейвлета
    
    Returns:
        watermarked_img (2d-array float64, C-совместимый): изображение с ЦВЗ
    """
    cdef bytes wt_name_bytes = wavelet_name.encode('ascii')
    cdef double h[8], g[8]
    cdef int L
    get_wavelet_filters(wt_name_bytes, h, g, &L) # Инициализация фильтров
    
    # Размеры изображения и подполос
    cdef int H = image.shape[0]
    cdef int W = image.shape[1]
    cdef int half_H = H >> 1
    cdef int half_W = W >> 1
    cdef int wm_len = watermark.shape[0]

    # Выделение буферов
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] LL = np.empty((half_H, half_W), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] LH = np.empty((half_H, half_W), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] HL = np.empty((half_H, half_W), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] HH = np.empty((half_H, half_W), dtype=np.float64)
    
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] temp_1d = np.empty(H * W, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_a = np.empty(half_W, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_d = np.empty(half_W, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_in = np.empty(H, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_a = np.empty(half_H, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_d = np.empty(half_H, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_out = np.empty(H, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_out = np.empty(W, dtype=np.float64)
    
    # void C-указатели на данные массивов для передачи в C-функции
    cdef double* temp_ptr = <double*>cnp.PyArray_DATA(temp_1d)
    cdef double* row_a_ptr = <double*>cnp.PyArray_DATA(row_a)
    cdef double* row_d_ptr = <double*>cnp.PyArray_DATA(row_d)
    cdef double* col_in_ptr = <double*>cnp.PyArray_DATA(col_in)
    cdef double* col_a_ptr = <double*>cnp.PyArray_DATA(col_a)
    cdef double* col_d_ptr = <double*>cnp.PyArray_DATA(col_d)
    cdef double* col_out_ptr = <double*>cnp.PyArray_DATA(col_out)
    cdef double* row_out_ptr = <double*>cnp.PyArray_DATA(row_out)
    
    # Прямое 2D DWT-преобразование для всего изображения
    dwt_2d_image(image, LL, LH, HL, HH, temp_ptr, row_a_ptr, row_d_ptr, col_in_ptr, col_a_ptr, col_d_ptr, h, g, L)
    
    cdef double[:, :] lh_view = LH # Встраивание ЦВЗ в подполосу LH
    cdef int blocks_h = half_H >> 2
    cdef int blocks_w = half_W >> 2
    cdef int b_idx = 0
    cdef int bi, bj, r, c
    cdef double c1, c2

    for bi in range(blocks_h):
        for bj in range(blocks_w):
            if b_idx >= wm_len:
                break # Когда вписали весь ЦВЗ - заканчиваем
            
            # Координаты центра блока 4x4 в подполосе LH
            r = (bi << 2) + 1
            c = (bj << 2) + 1
            # Пара коэффициентов для встраивания
            c1 = lh_view[r, c]
            c2 = lh_view[r + 1, c + 1]
            
            # Непосредственное встраивание
            if watermark[b_idx] == 1:
                if c1 <= c2:
                    lh_view[r, c] = c2 + margin
            else:
                if c1 >= c2:
                    lh_view[r + 1, c + 1] = c1 + margin
                    
            b_idx += 1
        if b_idx >= wm_len:
            break # Когда вписали весь ЦВЗ - заканчиваем

    # Обратное 2D DWT-преобразование для всего изображения
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] watermarked_img = np.empty((H, W), dtype=np.float64)
    idwt_2d_image(LL, LH, HL, HH, watermarked_img, temp_ptr, col_a_ptr, col_d_ptr, col_out_ptr, row_a_ptr, row_d_ptr, row_out_ptr, h, g, L)

    return watermarked_img

def extract_watermark_dwt(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] image, 
                          int wm_length,
                          str wavelet_name="sym4"):
    """
    Извлечение.
    
    Args:
        image (2d-array float64, C-совместимый): изображение с ЦВЗ
        wm_length (int): длина ЦВЗ
        wavelet_name (str): тип вейвлета
    
    Returns:
        extracted_wm (1D-array int32): извлечённый ЦВЗ
    """
    cdef bytes wt_name_bytes = wavelet_name.encode('ascii')
    cdef double h[8], g[8]
    cdef int L
    get_wavelet_filters(wt_name_bytes, h, g, &L) # Инициализация фильтров
    
    cdef int H = image.shape[0]
    cdef int W = image.shape[1]
    cdef int half_H = H >> 1
    cdef int half_W = W >> 1
    
    # Выделение буферов
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] LL = np.empty((half_H, half_W), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] LH = np.empty((half_H, half_W), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] HL = np.empty((half_H, half_W), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] HH = np.empty((half_H, half_W), dtype=np.float64)
    
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] temp_1d = np.empty(H * W, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_a = np.empty(half_W, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] row_d = np.empty(half_W, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_in = np.empty(H, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_a = np.empty(half_H, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] col_d = np.empty(half_H, dtype=np.float64)
    
    # void C-указатели на данные массивов для передачи в C-функции
    cdef double* temp_ptr = <double*>cnp.PyArray_DATA(temp_1d)
    cdef double* row_a_ptr = <double*>cnp.PyArray_DATA(row_a)
    cdef double* row_d_ptr = <double*>cnp.PyArray_DATA(row_d)
    cdef double* col_in_ptr = <double*>cnp.PyArray_DATA(col_in)
    cdef double* col_a_ptr = <double*>cnp.PyArray_DATA(col_a)
    cdef double* col_d_ptr = <double*>cnp.PyArray_DATA(col_d)
    
    # Прямое 2D DWT-преобразование для всего изображения
    dwt_2d_image(image, LL, LH, HL, HH, temp_ptr, row_a_ptr, row_d_ptr, col_in_ptr, col_a_ptr, col_d_ptr, h, g, L)
    
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] extracted_wm = np.zeros(wm_length, dtype=np.int32)
    cdef double[:, :] lh_view = LH # Извлечение ЦВЗ из подполосы LH
    cdef int blocks_h = half_H >> 2
    cdef int blocks_w = half_W >> 2
    cdef int b_idx = 0
    cdef int bi, bj, r, c
    cdef double c1, c2

    for bi in range(blocks_h):
        for bj in range(blocks_w):
            if b_idx >= wm_length:
                break
            
            r = (bi << 2) + 1
            c = (bj << 2) + 1
            c1 = lh_view[r, c]
            c2 = lh_view[r + 1, c + 1]
            
            # Извлечение бита ЦВЗ при сравнении коэффициентов
            extracted_wm[b_idx] = 1 if c1 > c2 else 0
            b_idx += 1
        if b_idx >= wm_length:
            break

    return extracted_wm

def _require_even(image):
    """
    Проверяет, что стороны изображения чётные.

    Одноуровневое разложение делит каждую ось пополам, и подполосы выделяются
    как H//2 x W//2. При нечётной стороне последняя строка или столбец не
    помещаются в буфер, и запись уходит за его границу: без этой проверки
    dwt_2d_image поднимает IndexError из середины цикла свёртки.

    Args:
        image (np.ndarray): канал яркости, форма (H, W)

    Returns:
        None

    Raises:
        ValueError: если хотя бы одна сторона нечётная
    """
    height, width = image.shape[0], image.shape[1]
    if height % 2 or width % 2:
        raise ValueError(
            f"Размер {width}x{height} имеет нечётную сторону, "
            f"одноуровневое DWT требует чётных. Обрежьте изображение."
        )


class DWT(Ready_Frequency_Embeddings):
    """
    Встраивание ЦВЗ в область дискретного вейвлет-преобразования.

    Изображение раскладывается на подполосы LL, LH, HL, HH одним уровнем
    вейвлет-преобразования. Бит кодируется в детализирующих подполосах,
    устойчивых к сжатию сильнее, чем отдельные коэффициенты DCT, за счёт
    того, что вейвлет-базис локализован и по частоте, и по пространству.
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
                margin (float): изменение коэффициентов при встраивании (по умолчанию 15.0)
                wavelet_name (str): тип вейвлета, 'haar', 'db4' или 'sym4' (по умолчанию 'sym4')

        Returns:
            None
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        watermark = bits_to_array(args["watermark_bits"])
        margin = float(args.get("margin", 15.0))
        wavelet_name = args.get("wavelet_name", "sym4")

        luma, chroma = load_luma(input_data)
        _require_even(luma)
        save_luma(
            embed_watermark_dwt(luma, watermark, margin, wavelet_name),
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
                wavelet_name (str): тип вейвлета (по умолчанию 'sym4').
                    Должен совпадать со значением при встраивании.

        Returns:
            str: строка из '0' и '1' длины num_bits
        """
        input_data = args["input_data"]
        num_bits = int(args["num_bits"])
        wavelet_name = args.get("wavelet_name", "sym4")

        luma, _ = load_luma(input_data)
        _require_even(luma)
        return array_to_bits(extract_watermark_dwt(luma, num_bits, wavelet_name))
