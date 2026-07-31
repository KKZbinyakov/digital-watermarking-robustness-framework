import numpy as np
cimport numpy as cnp
from libc.math cimport cos, sqrt, M_PI
from ...utils.embedding_utils import *

cnp.import_array()

# Используем C-массивы
ctypedef double DMatrix[8][8]
ctypedef double DBlock[8][8]

# Константа DCT-матрицы и флаг её вычисленности
cdef DMatrix _C_DCT
cdef bint _C_DCT_INITIALIZED = False

cdef void init_dct_matrix():
    """
    Вычисляет ортогональную матрицу DCT 8x8 один раз при первом вызове.
    """
    global _C_DCT, _C_DCT_INITIALIZED
    if _C_DCT_INITIALIZED:
        return
    
    cdef int k, n
    cdef double alpha, factor = sqrt(2.0/8.0)  # factor = 0.5
    
    for k in range(8):
        alpha = 1.0/sqrt(2.0) if k == 0 else 1.0
        for n in range(8):
            _C_DCT[k][n] = alpha * factor * cos(M_PI * (2.0*n+1.0) * k / 16.0)
    
    _C_DCT_INITIALIZED = True

cdef void apply_dct_8x8(const double block_img[8][8], double block_dct[8][8], const double C[8][8]):
    """
    Прямое DCT-преобразование: F = C * f * C^T, где C - матрица DCT, f - блок пикселей
    """
    cdef double temp[8][8]
    cdef int i, j, k
    cdef double sum_val
    
    # Умножение C * block_img, получение temp
    for i in range(8):
        for j in range(8):
            sum_val = 0.0
            for k in range(8):
                sum_val += C[i][k] * block_img[k][j]
            temp[i][j] = sum_val
            
    # Умножение temp * C^T, получение блока частот
    for i in range(8):
        for j in range(8):
            sum_val = 0.0
            for k in range(8):
                sum_val += temp[i][k] * C[j][k]
            block_dct[i][j] = sum_val

cdef void apply_idct_8x8(const double block_dct[8][8], double block_idct[8][8], const double C[8][8]):
    """
    Обратное DCT-преобразование: f = C^T * F * C
    """
    cdef double temp[8][8]
    cdef int i, j, k
    cdef double sum_val
    
    # Умножение C^T * block_dct, получение temp
    for i in range(8):
        for j in range(8):
            sum_val = 0.0
            for k in range(8):
                sum_val += C[k][i] * block_dct[k][j]
            temp[i][j] = sum_val
            
    # Умножение temp * C, получение блока пикселей
    for i in range(8):
        for j in range(8):
            sum_val = 0.0
            for k in range(8):
                sum_val += temp[i][k] * C[k][j]
            block_idct[i][j] = sum_val

def embed_watermark_dct(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] image, 
                        cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] watermark, 
                        double margin=5.0):
    """
    Встраивание ЦВЗ.
    
    Args:
        image (2d-array float64, C-совместимый): чистое изображение
        watermark (1D-array int32): ЦВЗ
        margin (double): изменение коэффициентов при непосредственном встраивании
    
    Returns:
        watermarked_img (2d-array float64, C-совместимый): изображение с ЦВЗ
    """
    init_dct_matrix()
    
    cdef int H = image.shape[0]
    cdef int W = image.shape[1]
    cdef int blocks_h = H // 8
    cdef int blocks_w = W // 8
    cdef int wm_len = watermark.shape[0] # Длина ЦВЗ

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] watermarked_img = image.copy() # Копия массива пикселей
    cdef double[:, :] img_view = watermarked_img # Преобразование к memoryview
    
    cdef DBlock block_img, block_dct, block_idct_arr
    
    cdef int b_idx = 0
    cdef int bi, bj, r, c
    cdef double c1, c2

    for bi in range(blocks_h):
        for bj in range(blocks_w):
            if b_idx >= wm_len: # Когда вписали весь ЦВЗ - заканчиваем
                break
                
            # Копируем блок 8x8 из изображения в локальный C-массив
            for r in range(8):
                for c in range(8):
                    block_img[r][c] = img_view[bi*8 + r, bj*8 + c]
            
            # Применяем прямое DCT
            apply_dct_8x8(block_img, block_dct, _C_DCT)
            
            # Выбираем среднечастотные коэффициенты для модификации
            c1 = block_dct[2][3]
            c2 = block_dct[3][2]
            
            # Непосредственное встраивание
            if watermark[b_idx] == 1:
                if c1 <= c2:
                    block_dct[2][3] = c2 + margin
            else:
                if c1 >= c2:
                    block_dct[3][2] = c1 + margin
                    
            # Применяем обратное DCT
            apply_idct_8x8(block_dct, block_idct_arr, _C_DCT)
            
            # Записываем модифицированный блок обратно
            for r in range(8):
                for c in range(8):
                    img_view[bi*8 + r, bj*8 + c] = block_idct_arr[r][c]
                    
            b_idx += 1
        if b_idx >= wm_len: # Когда вписали весь ЦВЗ - заканчиваем
            break

    return watermarked_img

def extract_watermark_dct(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] image, 
                        int wm_length):
    """
    Извлечение ЦВЗ.
    
    Args:
        image (2D-array float64, C-совместимый): изображение с ЦВЗ
        wm_length (int): длина ЦВЗ
    
    Returns:
        extracted_wm (1D-array int32): извлечённый ЦВЗ
    """
    init_dct_matrix()
    
    cdef int H = image.shape[0]
    cdef int W = image.shape[1]
    cdef int blocks_h = H // 8
    cdef int blocks_w = W // 8
    cdef int total_blocks = blocks_h * blocks_w
    
    if wm_length > total_blocks: # Если поданная длина ЦВЗ больше, чем максимально возможное
        raise ValueError(f"Задано {wm_length} бит, но в изображении только {total_blocks} блоков 8x8.")

    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode='c'] extracted_wm = np.zeros(wm_length, dtype=np.int32) # Будущий ЦВЗ
    
    cdef DBlock block_img, block_dct
    
    cdef int b_idx = 0
    cdef int bi, bj, r, c
    cdef double c1, c2
    cdef double[:, :] img_view = image

    for bi in range(blocks_h):
        for bj in range(blocks_w):
            if b_idx >= wm_length:
                break
                
            for r in range(8):
                for c in range(8):
                    block_img[r][c] = img_view[bi*8 + r, bj*8 + c]
            
            apply_dct_8x8(block_img, block_dct, _C_DCT)
            
            c1 = block_dct[2][3]
            c2 = block_dct[3][2]
            
            # Извлечение бита ЦВЗ при сравнении коэффициентов
            extracted_wm[b_idx] = 1 if c1 > c2 else 0
            
            b_idx += 1
        if b_idx >= wm_length:
            break

    return extracted_wm

class DCT(Ready_Frequency_Embeddings):
    """
    Встраивание ЦВЗ в область дискретного косинусного преобразования.

    Изображение режется на блоки 8x8, каждый блок несёт один бит. Бит
    кодируется порядком пары среднечастотных коэффициентов (2, 3) и (3, 2):
    низкие частоты трогать нельзя из-за заметности, высокие уничтожаются
    сжатием. Ёмкость равна числу целых блоков 8x8.
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
                margin (float): изменение коэффициентов при встраивании (по умолчанию 5.0)

        Returns:
            None
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        watermark = bits_to_array(args["watermark_bits"])
        margin = float(args.get("margin", 5.0))

        luma, chroma = load_luma(input_data)
        save_luma(embed_watermark_dct(luma, watermark, margin), chroma, output_data)

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

        Returns:
            str: строка из '0' и '1' длины num_bits
        """
        input_data = args["input_data"]
        num_bits = int(args["num_bits"])

        luma, _ = load_luma(input_data)
        return array_to_bits(extract_watermark_dct(luma, num_bits))

    @staticmethod
    def capacity(args: dict = {
        "input_data": None
    }):
        """
        Максимальная длина ЦВЗ для изображения: по одному биту на блок 8x8.

        Args:
            args (dict): параметры расчёта
                input_data (str): путь к изображению

        Returns:
            int: ёмкость в битах
        """
        luma, _ = load_luma(args["input_data"])
        return (luma.shape[0] // 8) * (luma.shape[1] // 8)
