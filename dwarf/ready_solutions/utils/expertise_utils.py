"""Вспомогательные функции метрик экспертизы.

Здесь лежит то, что нужно нескольким метрикам сразу: приведение матриц
изображений к нужному виду, разделимая фильтрация и карты SSIM, фазовая
согласованность и градиент для FSIM, работа с битовыми строками и метками
детектора, а также ленивое создание нейросетевых метрик pyiqa.
"""

import numpy as np


def to_rgb_float(image) -> np.ndarray:
    """
    Приводит матрицу изображения к вещественному RGB.

    Тип меняется на float64, потому что метрики считают разности и квадраты:
    на uint8 такая арифметика переполняется. Значения не округляются и не
    прижимаются к диапазону — метрика измеряет то, что ей дали.

    Args:
        image (np.ndarray): матрица формы (H, W, 3) любого числового типа

    Returns:
        np.ndarray: матрица формы (H, W, 3) типа float64

    Raises:
        ValueError: если матрица не имеет формы (H, W, 3)
    """
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an RGB image matrix of shape (H, W, 3), got {array.shape}")
    return array


def to_gray(image) -> np.ndarray:
    """
    Приводит матрицу изображения к вещественному каналу яркости.

    Яркость считается по коэффициентам ITU-R BT.601 без округления до целых
    уровней: эталонные реализации SSIM, MS-SSIM, FSIM и VIF работают именно с
    вещественной яркостью, и промежуточное округление сместило бы значения.

    Матрица (H, W) принимается как уже готовая яркость: изображения-знаки
    часто хранятся одноканальными.

    Args:
        image (np.ndarray): матрица формы (H, W, 3) или (H, W) любого числового типа

    Returns:
        np.ndarray: матрица формы (H, W) типа float64

    Raises:
        ValueError: если матрица не имеет формы (H, W, 3) или (H, W)
    """
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 3:
        return 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    raise ValueError(f"expected an image matrix of shape (H, W, 3) or (H, W), got {array.shape}")


def gauss1d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """
    Строит одномерное нормированное гауссово ядро.

    Args:
        size (int): длина ядра
        sigma (float): стандартное отклонение

    Returns:
        np.ndarray: массив длины size типа float64, сумма равна единице
    """
    positions = np.arange(size) - size // 2
    kernel = np.exp(-(positions**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def sepfilter(image: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    """
    Разделимая свёртка изображения одномерным ядром.

    Режим 'same' дополняет края нулями и сохраняет размер. Режим 'valid'
    оставляет только те отсчёты, для которых ядро целиком помещается внутрь
    изображения, уменьшая размер на длину ядра без единицы.

    Args:
        image (np.ndarray): изображение (H, W) типа float64
        kernel (np.ndarray): одномерное ядро
        mode (str): 'same' или 'valid'

    Returns:
        np.ndarray: отфильтрованное изображение типа float64

    Raises:
        ValueError: если режим неизвестен
    """
    if mode not in ("same", "valid"):
        raise ValueError(f"unknown mode={mode!r}, expected 'same' or 'valid'")
    rows = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode=mode), 1, image)
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode=mode), 0, rows)


def ssim_maps(first: np.ndarray, second: np.ndarray, kernel: np.ndarray = None) -> tuple:
    """
    Считает карту SSIM и карту контраст-структуры для пары яркостей.

    Карта контраст-структуры возвращается отдельно, поскольку MS-SSIM использует
    её на всех масштабах, кроме последнего.

    Свёртка идёт в режиме 'valid', как в эталонной реализации Ванга: при
    дополнении краёв в окно попадают несуществующие пиксели, и краевая полоса
    шириной в половину окна systematically завышает или занижает индекс.
    Поэтому карты меньше входа на длину окна без единицы по каждой оси.

    Args:
        first (np.ndarray): яркость первого изображения (H, W)
        second (np.ndarray): яркость второго изображения (H, W)
        kernel (np.ndarray): одномерное окно, по умолчанию гауссово 11 на 1.5

    Returns:
        tuple: (ssim_map, cs_map), оба массива формы (H - k + 1, W - k + 1)

    Raises:
        ValueError: если изображение меньше окна
    """
    if kernel is None:
        kernel = gauss1d(11, 1.5)
    if min(first.shape) < kernel.shape[0]:
        raise ValueError(f"image {first.shape} is smaller than the window {kernel.shape[0]}: SSIM is undefined")
    stabilizer_l = (0.01 * 255) ** 2
    stabilizer_c = (0.03 * 255) ** 2

    mean_first = sepfilter(first, kernel, "valid")
    mean_second = sepfilter(second, kernel, "valid")
    mean_first_sq = mean_first**2
    mean_second_sq = mean_second**2
    mean_product = mean_first * mean_second

    var_first = sepfilter(first * first, kernel, "valid") - mean_first_sq
    var_second = sepfilter(second * second, kernel, "valid") - mean_second_sq
    covariance = sepfilter(first * second, kernel, "valid") - mean_product

    ssim_map = ((2 * mean_product + stabilizer_l) * (2 * covariance + stabilizer_c)) / (
        (mean_first_sq + mean_second_sq + stabilizer_l) * (var_first + var_second + stabilizer_c)
    )
    cs_map = (2 * covariance + stabilizer_c) / (var_first + var_second + stabilizer_c)
    return ssim_map, cs_map


def downsample_by_two(image: np.ndarray) -> np.ndarray:
    """
    Усредняет изображение блоком 2 на 2 и прореживает вдвое.

    Повторяет понижение масштаба из эталонной реализации MS-SSIM: свёртка с
    ядром из четырёх равных весов при симметричном дополнении краёв, затем
    выборка каждого второго отсчёта.

    Args:
        image (np.ndarray): изображение (H, W) типа float64

    Returns:
        np.ndarray: изображение примерно вдвое меньшего размера
    """
    padded = np.pad(image, ((1, 0), (1, 0)), mode="symmetric")
    averaged = (padded[:-1, :-1] + padded[:-1, 1:] + padded[1:, :-1] + padded[1:, 1:]) / 4.0
    return averaged[::2, ::2]


SCHARR_X = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=np.float64) / 16.0
"""Горизонтальное ядро Шарра, как в эталонной реализации FSIM."""


def conv2(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Двумерная корреляция с симметричным дополнением краёв, режим 'same'.

    Args:
        image (np.ndarray): изображение (H, W) типа float64
        kernel (np.ndarray): двумерное ядро

    Returns:
        np.ndarray: результат (H, W) типа float64
    """
    kernel_height, kernel_width = kernel.shape
    padded = np.pad(image, ((kernel_height // 2,) * 2, (kernel_width // 2,) * 2), mode="symmetric")
    result = np.zeros_like(image, dtype=np.float64)
    for row in range(kernel_height):
        for column in range(kernel_width):
            result += kernel[row, column] * padded[row : row + image.shape[0], column : column + image.shape[1]]
    return result


def scharr_gm(image: np.ndarray) -> np.ndarray:
    """
    Модуль градиента, посчитанный оператором Шарра.

    Args:
        image (np.ndarray): изображение (H, W) типа float64

    Returns:
        np.ndarray: модуль градиента (H, W) типа float64
    """
    horizontal = conv2(image, SCHARR_X)
    vertical = conv2(image, SCHARR_X.T)
    return np.sqrt(horizontal**2 + vertical**2)


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Усредняет изображение блоком factor на factor и прореживает его.

    Args:
        image (np.ndarray): изображение (H, W) типа float64
        factor (int): коэффициент прореживания, при значении 1 и меньше возвращается вход

    Returns:
        np.ndarray: прореженное изображение
    """
    if factor <= 1:
        return image
    kernel = np.ones((factor, factor)) / (factor * factor)
    return conv2(image, kernel)[::factor, ::factor]


def rgb_to_yiq(rgb: np.ndarray) -> tuple:
    """
    Переводит RGB в цветовое пространство YIQ.

    Args:
        rgb (np.ndarray): изображение (H, W, 3) со значениями 0..255

    Returns:
        tuple: три массива (H, W) — каналы Y, I и Q
    """
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    in_phase = 0.596 * red - 0.274 * green - 0.322 * blue
    quadrature = 0.211 * red - 0.523 * green + 0.312 * blue
    return luma, in_phase, quadrature


def freq_grid(rows: int, cols: int) -> tuple:
    """
    Строит частотную сетку для банка фильтров log-Gabor.

    Радиус и углы сдвигаются функцией ifftshift, чтобы совпадать с раскладкой
    спектра numpy. Нулевая частота принудительно ставится в единицу: в неё
    делят при построении фильтров.

    Args:
        rows (int): число строк изображения
        cols (int): число столбцов изображения

    Returns:
        tuple: (radius, sin(theta), cos(theta)), три массива формы (rows, cols)
    """
    if cols % 2:
        x_range = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
        x_range = np.arange(-cols / 2, cols / 2) / cols
    if rows % 2:
        y_range = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
        y_range = np.arange(-rows / 2, rows / 2) / rows

    x_grid, y_grid = np.meshgrid(x_range, y_range)
    radius = np.fft.ifftshift(np.sqrt(x_grid**2 + y_grid**2))
    theta = np.fft.ifftshift(np.arctan2(-y_grid, x_grid))
    radius[0, 0] = 1
    return radius, np.sin(theta), np.cos(theta)


def phase_congruency(
    image: np.ndarray,
    nscale: int = 4,
    norient: int = 4,
    min_wavelength: float = 6,
    mult: float = 2.0,
    sigma_onf: float = 0.55,
    dtheta_on_sigma: float = 1.2,
    noise_k: float = 2.0,
    cutoff: float = 0.5,
    gain: float = 10.0,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Считает фазовую согласованность по Ковеси (вариант phasecong2 из FSIM).

    Энергия берётся по фазовой девиации, взвешивается мерой частотного разброса
    и обрезается рэлеевским порогом шума, после чего по ориентациям выполняется
    моментный анализ и возвращается максимальный момент.

    Args:
        image (np.ndarray): яркость (H, W) типа float64
        nscale (int): число масштабов банка фильтров
        norient (int): число ориентаций
        min_wavelength (float): длина волны самого мелкого масштаба
        mult (float): множитель длины волны между масштабами
        sigma_onf (float): относительная полоса пропускания фильтра
        dtheta_on_sigma (float): отношение шага ориентаций к их разбросу
        noise_k (float): множитель стандартного отклонения при пороге шума
        cutoff (float): порог доли частотного разброса
        gain (float): крутизна весовой сигмоиды
        eps (float): стабилизатор деления

    Returns:
        np.ndarray: карта фазовой согласованности (H, W)
    """
    rows, cols = image.shape
    spectrum = np.fft.fft2(image)
    radius, sintheta, costheta = freq_grid(rows, cols)
    low_pass = 1.0 / (1.0 + (radius / 0.45) ** (2 * 15))

    log_gabors = []
    for scale in range(nscale):
        center = 1.0 / (min_wavelength * (mult**scale))
        gabor = np.exp(-((np.log(radius / center)) ** 2) / (2 * np.log(sigma_onf) ** 2)) * low_pass
        gabor[0, 0] = 0
        log_gabors.append(gabor)

    cov_xx = np.zeros((rows, cols))
    cov_yy = np.zeros((rows, cols))
    cov_xy = np.zeros((rows, cols))
    theta_sigma = np.pi / norient / dtheta_on_sigma

    for orientation in range(norient):
        angle = orientation * np.pi / norient
        delta_sin = sintheta * np.cos(angle) - costheta * np.sin(angle)
        delta_cos = costheta * np.cos(angle) + sintheta * np.sin(angle)
        spread = np.exp(-(np.abs(np.arctan2(delta_sin, delta_cos)) ** 2) / (2 * theta_sigma**2))

        responses = []
        sum_even = np.zeros((rows, cols))
        sum_odd = np.zeros((rows, cols))
        sum_amplitude = np.zeros((rows, cols))
        max_amplitude = np.zeros((rows, cols))
        for scale in range(nscale):
            response = np.fft.ifft2(spectrum * log_gabors[scale] * spread)
            responses.append(response)
            amplitude = np.abs(response)
            sum_even += response.real
            sum_odd += response.imag
            sum_amplitude += amplitude
            max_amplitude = np.maximum(max_amplitude, amplitude)

        total_energy = np.sqrt(sum_even**2 + sum_odd**2) + eps
        mean_even = sum_even / total_energy
        mean_odd = sum_odd / total_energy
        energy = np.zeros((rows, cols))
        for response in responses:
            even, odd = response.real, response.imag
            energy += even * mean_even + odd * mean_odd - np.abs(even * mean_odd - odd * mean_even)

        tau = np.median(sum_amplitude) / np.sqrt(np.log(4))
        total_tau = tau * (1 - (1 / mult) ** nscale) / (1 - 1 / mult)
        threshold = total_tau * np.sqrt(np.pi / 2) + noise_k * total_tau * np.sqrt((4 - np.pi) / 2)
        energy = np.maximum(energy - threshold, 0)

        width = (sum_amplitude / (max_amplitude + eps) - 1) / (nscale - 1)
        weight = 1.0 / (1 + np.exp((cutoff - width) * gain))
        congruency = weight * energy / (sum_amplitude + eps)

        cov_xx += (congruency * np.cos(angle)) ** 2
        cov_yy += (congruency * np.sin(angle)) ** 2
        cov_xy += congruency * np.cos(angle) * congruency * np.sin(angle)

    cov_xx /= norient / 2
    cov_yy /= norient / 2
    cov_xy = 4 * cov_xy / norient
    spread_term = np.sqrt(cov_xy**2 + (cov_xx - cov_yy) ** 2) + eps
    return (cov_yy + cov_xx + spread_term) / 2


def fsim_luma_maps(first_luma: np.ndarray, second_luma: np.ndarray) -> tuple:
    """
    Считает карту сходства по яркости и карту максимума фазовой согласованности.

    Обе яркости должны быть уже прорежены одним и тем же коэффициентом.

    Args:
        first_luma (np.ndarray): яркость первого изображения (H, W)
        second_luma (np.ndarray): яркость второго изображения (H, W)

    Returns:
        tuple: (карта сходства (H, W), карта весов (H, W))
    """
    congruency_first = phase_congruency(first_luma)
    congruency_second = phase_congruency(second_luma)
    gradient_first = scharr_gm(first_luma)
    gradient_second = scharr_gm(second_luma)

    stabilizer_pc, stabilizer_gm = 0.85, 160.0
    similarity_pc = (2 * congruency_first * congruency_second + stabilizer_pc) / (
        congruency_first**2 + congruency_second**2 + stabilizer_pc
    )
    similarity_gm = (2 * gradient_first * gradient_second + stabilizer_gm) / (
        gradient_first**2 + gradient_second**2 + stabilizer_gm
    )
    return similarity_pc * similarity_gm, np.maximum(congruency_first, congruency_second)


def align_bits(original_bits: str, extracted_bits: str, allow_length_mismatch: bool) -> tuple:
    """
    Приводит пару битовых строк к общей длине, контролируя расхождение длин.

    По умолчанию расхождение длин считается ошибкой. Молчаливое сравнение по
    минимальной длине опасно для бенчмарка: если извлечение вернуло меньше бит,
    чем встраивалось, метрика посчитается по обрезку и в отчёте будет неотличима
    от честного замера.

    Args:
        original_bits (str): исходный ЦВЗ, строка из символов '0' и '1'
        extracted_bits (str): извлечённый ЦВЗ, строка из символов '0' и '1'
        allow_length_mismatch (bool): разрешить сравнение по минимальной длине

    Returns:
        tuple: (original_bits, extracted_bits, length) — обрезанные строки и их длина

    Raises:
        ValueError: если длины различаются, а allow_length_mismatch не установлен
    """
    if len(original_bits) != len(extracted_bits) and not allow_length_mismatch:
        raise ValueError(
            f"bit string lengths differ: {len(original_bits)} vs {len(extracted_bits)}. "
            f"This usually means extraction returned fewer bits than were embedded. "
            f"To compare the common prefix deliberately, pass allow_length_mismatch=True"
        )
    length = min(len(original_bits), len(extracted_bits))
    return original_bits[:length], extracted_bits[:length], length


def bits_to_array(bits: str) -> np.ndarray:
    """
    Переводит битовую строку в массив нулей и единиц.

    Сравнивать ЦВЗ поэлементно можно только в виде массива: у строк Python
    оператор != даёт одно логическое значение на всю строку, и метрика,
    посчитанная на таком сравнении, молча теряет число ошибок.

    Args:
        bits (str): строка из символов '0' и '1'

    Returns:
        np.ndarray: массив длины len(bits) типа uint8 со значениями 0 и 1

    Raises:
        ValueError: если строка содержит символы, отличные от '0' и '1'
    """
    codes = np.frombuffer(bits.encode("ascii"), dtype=np.uint8)
    allowed = (codes == ord("0")) | (codes == ord("1"))
    if not allowed.all():
        position = int(np.argmin(allowed))
        raise ValueError(f"bit string must contain only '0' and '1', got {bits[position]!r} at position {position}")
    return (codes == ord("1")).astype(np.uint8)


def bits_to_pm1(bits: str) -> np.ndarray:
    """
    Переводит битовую строку в массив из значений -1 и +1.

    Args:
        bits (str): строка из символов '0' и '1'

    Returns:
        np.ndarray: массив длины len(bits) типа float64

    Raises:
        ValueError: если строка содержит символы, отличные от '0' и '1'
    """
    return np.where(bits_to_array(bits) == 1, 1.0, -1.0)


def avg_ranks(values: np.ndarray) -> np.ndarray:
    """
    Возвращает ранги значений, усредняя ранги внутри групп совпадающих.

    Усреднение обязательно для AUC: без него совпадающие оценки детектора
    получили бы произвольный порядок, и результат зависел бы от него.

    Args:
        values (np.ndarray): одномерный массив значений

    Returns:
        np.ndarray: массив рангов той же длины, нумерация с единицы
    """
    values = np.asarray(values, dtype=float)
    count = values.shape[0]
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]

    starts_group = np.empty(count, dtype=bool)
    starts_group[0] = True
    starts_group[1:] = ordered[1:] != ordered[:-1]

    group_start = np.flatnonzero(starts_group)
    group_end = np.append(group_start[1:], count)
    group_index = np.cumsum(starts_group) - 1

    ranks = np.empty(count, dtype=float)
    ranks[order] = ((group_start + 1 + group_end) / 2.0)[group_index]
    return ranks


def confusion_counts(y_true, y_pred) -> tuple:
    """
    Считает элементы матрицы ошибок бинарного детектора.

    Args:
        y_true: истинные метки, последовательность из 0 и 1
        y_pred: предсказанные метки, последовательность из 0 и 1

    Returns:
        tuple: (tp, tn, fp, fn) — целые числа

    Raises:
        ValueError: если длины меток не совпадают или массивы пусты
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"label shapes differ: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("label arrays are empty")

    positive_true = y_true == 1
    positive_pred = y_pred == 1
    return (
        int((positive_pred & positive_true).sum()),
        int((~positive_pred & ~positive_true).sum()),
        int((positive_pred & ~positive_true).sum()),
        int((~positive_pred & positive_true).sum()),
    )


_IQA_CACHE = {}
"""Кэш нейросетевых метрик pyiqa: создание метрики тянет веса и заметно медленное."""


def to_tensor(image):
    """
    Приводит матрицу изображения к тензору torch формы (1, 3, H, W) в диапазоне [0, 1].

    Args:
        image (np.ndarray): матрица формы (H, W, 3) со значениями 0..255

    Returns:
        torch.Tensor: тензор формы (1, 3, H, W) типа float32

    Raises:
        ValueError: если матрица не имеет формы (H, W, 3)
        RuntimeError: если пакет torch не установлен
    """
    array = np.ascontiguousarray(to_rgb_float(image), dtype=np.float32) / 255.0
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Neural metrics require torch: pip install pyiqa") from error
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def iqa_metric(name: str):
    """
    Лениво создаёт и кэширует метрику pyiqa по её имени.

    Создание вынесено из импорта модуля, чтобы отсутствие pyiqa не ломало импорт
    пакета и не выбивало метрику из реестра.

    Args:
        name (str): имя метрики в pyiqa, например 'lpips' или 'niqe'

    Returns:
        Метрика pyiqa, готовая к вызову

    Raises:
        RuntimeError: если пакет pyiqa не установлен
    """
    if name not in _IQA_CACHE:
        try:
            import pyiqa  # noqa: F401
        except ImportError as error:
            raise RuntimeError(f"Metric {name} requires the pyiqa package: pip install pyiqa") from error
        _IQA_CACHE[name] = pyiqa.create_metric(name)
    return _IQA_CACHE[name]
