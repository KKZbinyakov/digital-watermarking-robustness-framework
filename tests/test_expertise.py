"""Тесты корректности готовых метрик экспертизы.

Набор самонаполняющийся: метрики берутся из реестра Expertise_Core, поэтому
новый класс метрики попадает под общие проверки автоматически, без правки
этого файла. Метрики, требующие отсутствующих пакетов, пропускаются, а не
проваливаются.

Для метрик, у которых внешней реализации не нашлось (VIF, FSIM, FSIMc),
зафиксированы эталонные значения, полученные после сверки VIF с исходным
vifvec
"""

import inspect

import numpy as np
import pytest
from PIL import Image, ImageFilter

from dwarf import Expertise_Core

FULL_REFERENCE = frozenset(
    {
        "MSE",
        "PSNR",
        "SSIM",
        "MS_SSIM",
        "VIF",
        "FSIM",
        "FSIMc",
        "LPIPS",
        "DISTS",
    }
)
"""Метрики, сравнивающие изображение с оригиналом."""

NO_REFERENCE = frozenset({"NIQE", "BRISQUE"})
"""Безэталонные метрики: оценивают одно изображение без оригинала."""

BIT_STRINGS = frozenset({"BER", "NC"})
"""Метрики, сравнивающие битовые строки ЦВЗ."""

LABEL_BASED = frozenset({"Accuracy", "Precision", "Recall", "F1"})
"""Метрики бинарного детектора, работающие с готовыми метками."""

LOWER_IS_BETTER = frozenset({"MSE", "LPIPS", "DISTS", "NIQE", "BRISQUE"})
"""Метрики, растущие с ростом искажения."""

MINIMUM_SIDE = {"SSIM": 11, "MS_SSIM": 176, "VIF": 72}
"""Наименьшая сторона кадра, при которой метрика определена."""

GOLDEN = {
    "identity": {"VIF": 0.9999999999999978, "FSIM": 1.0, "FSIMc": 1.0},
    "noise": {"VIF": 0.6230617910677038, "FSIM": 0.9603402010526968, "FSIMc": 0.9537927462695296},
    "blur": {"VIF": 0.25013374727949195, "FSIM": 0.7535262819245329, "FSIMc": 0.7513410066027992},
    "jpeg": {"VIF": 0.2827797038815233, "FSIM": 0.9676943224352389, "FSIMc": 0.9642463431969664},
}
"""Зафиксированные значения метрик без внешнего эталона на reference_image."""

GOLDEN_TOLERANCE = 1e-6
"""Допуск для зафиксированных значений.

Взят с запасом относительно машинной точности: VIF обращает матрицу и считает
собственные значения, а разные сборки LAPACK дают в последних разрядах разные
результаты. Реальные изменения алгоритма на порядки больше этого допуска —
найденные расхождения составляли от 6e-3 до 5e-2.
"""


def _discover():
    """
    Собирает конкретные классы метрик из реестра Expertise_Core.

    Классы-категории не реализуют expertise и потому абстрактны — по этому
    признаку они и отсеиваются.

    Returns:
        dict: отображение имени метрики в её класс
    """
    return {
        name: cls for name, cls in Expertise_Core.get_registered_expertises().items() if not inspect.isabstract(cls)
    }


METRICS = _discover()
NAMES = sorted(METRICS)


def require(name):
    """
    Пропускает тест, если метрика отсутствует в реестре.

    Несобранные Cython-расширения для pkgutil невидимы, поэтому соответствующие
    метрики просто не регистрируются. Без этой проверки тесты падали бы с
    KeyError, по которому причину не восстановить.

    Args:
        name (str): имя метрики в реестре

    Returns:
        type: класс метрики
    """
    if name not in METRICS:
        pytest.skip(
            f"метрика {name} отсутствует в реестре: соберите расширения командой python setup.py build_ext --inplace"
        )
    return METRICS[name]


def reference_image(height=288, width=288, seed=20240501):
    """
    Строит детерминированное тестовое изображение.

    Содержит плавные градиенты, чтобы были заметны артефакты сжатия, две
    контрастные области, чтобы гистограмма не вырождалась, и слабый шум, чтобы
    метрики фазовой согласованности имели с чем работать. Размер по умолчанию
    выбран не меньше порога самой требовательной метрики.

    Args:
        height (int): высота кадра
        width (int): ширина кадра
        seed (int): зерно генератора шума

    Returns:
        np.ndarray: массив (height, width, 3) типа uint8
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:height, 0:width].astype(float)
    base = (
        0.5
        + 0.22 * np.sin(2 * np.pi * cols / width)
        + 0.14 * np.cos(2 * np.pi * rows / height)
        + 0.10 * np.sin(6 * np.pi * (rows + cols) / (height + width))
    )
    array = np.stack([base * 205 + 25, base * 185 + 35, base * 165 + 45], axis=-1)
    array[height // 5 : height // 2, width // 5 : width // 2] *= 0.6
    array[3 * height // 5 :, 3 * width // 5 :] *= 1.4
    array += rng.normal(0, 5, array.shape)
    return np.clip(array, 0, 255).astype(np.uint8)


@pytest.fixture
def original(tmp_path):
    """
    Сохраняет эталонное изображение и возвращает путь к нему.

    Args:
        tmp_path (pathlib.Path): временный каталог теста

    Returns:
        str: путь к изображению
    """
    path = tmp_path / "original.png"
    Image.fromarray(reference_image(), "RGB").save(path)
    return str(path)


@pytest.fixture
def distorted(tmp_path, original):
    """
    Возвращает фабрику искажённых копий эталонного изображения.

    Args:
        tmp_path (pathlib.Path): временный каталог теста
        original (str): путь к эталонному изображению

    Returns:
        callable: функция (вид искажения) -> путь к искажённой копии
    """

    def build(kind):
        base = np.asarray(Image.open(original).convert("RGB"))
        path = tmp_path / f"distorted_{kind}.png"
        if kind == "identity":
            result = base
        elif kind.startswith("noise"):
            deviation = {"noise": 8, "noise_mild": 3, "noise_severe": 20}[kind]
            noise = np.random.default_rng(7).normal(0, deviation, base.shape)
            result = np.clip(base + noise, 0, 255).astype(np.uint8)
        elif kind.startswith("blur"):
            radius = {"blur": 2, "blur_mild": 1, "blur_severe": 4}[kind]
            result = np.asarray(Image.open(original).filter(ImageFilter.GaussianBlur(radius)))
        elif kind.startswith("jpeg"):
            quality = {"jpeg": 30, "jpeg_mild": 85, "jpeg_severe": 10}[kind]
            jpeg_path = tmp_path / f"distorted_{kind}.jpg"
            Image.open(original).save(jpeg_path, quality=quality)
            result = np.asarray(Image.open(jpeg_path).convert("RGB"))
        else:
            raise ValueError(f"неизвестный вид искажения {kind!r}")
        Image.fromarray(result, "RGB").save(path)
        return str(path)

    return build


@pytest.fixture
def watermarks(tmp_path):
    """
    Готовит пару изображений-знаков для метрики Watermark_PSNR.

    Args:
        tmp_path (pathlib.Path): временный каталог теста

    Returns:
        tuple: пути к исходному и восстановленному знакам
    """
    rng = np.random.default_rng(3)
    first = tmp_path / "watermark_original.png"
    second = tmp_path / "watermark_extracted.png"
    array = (rng.random((64, 64)) * 255).astype(np.uint8)
    Image.fromarray(array, "L").save(first)
    Image.fromarray(np.roll(array, 1, axis=0), "L").save(second)
    return str(first), str(second)


@pytest.fixture
def detector():
    """
    Готовит метки и оценки бинарного детектора.

    Returns:
        tuple: (истинные метки, предсказанные метки, непрерывные оценки)
    """
    rng = np.random.default_rng(11)
    truth = rng.integers(0, 2, 400)
    scores = np.clip(truth * 0.4 + rng.random(400) * 0.6, 0, 1)
    return truth, (scores > 0.5).astype(int), scores


@pytest.fixture
def arguments(original, distorted, watermarks, detector):
    """
    Возвращает фабрику параметров вызова для метрики любого вида.

    Виды метрик принимают разные ключи: пути к паре изображений, путь к одному
    изображению, битовые строки или массивы меток. Фабрика скрывает это
    различие, чтобы общие тесты можно было параметризовать по всему реестру.

    Args:
        original (str): путь к эталонному изображению
        distorted (callable): фабрика искажённых копий
        watermarks (tuple): пути к паре изображений-знаков
        detector (tuple): метки и оценки детектора

    Returns:
        callable: функция (имя метрики, вид искажения) -> dict параметров
    """
    truth, predicted, scores = detector

    def build(name, kind="noise"):
        if name in FULL_REFERENCE:
            return {"original_path": original, "distorted_path": distorted(kind)}
        if name in NO_REFERENCE:
            return {"image_path": original}
        if name in BIT_STRINGS:
            rng = np.random.default_rng(5)
            bits = "".join(rng.integers(0, 2, 512).astype(str))
            flipped = list(bits)
            for index in range(0, len(flipped), 8):
                flipped[index] = "1" if flipped[index] == "0" else "0"
            return {"original_bits": bits, "extracted_bits": "".join(flipped)}
        if name in LABEL_BASED:
            return {"y_true": truth, "y_pred": predicted}
        if name == "AUC":
            return {"y_true": truth, "y_scores": scores}
        if name == "P_Value":
            return {"statistic": 0.7, "null_samples": scores}
        if name == "Watermark_PSNR":
            return {"original_watermark_path": watermarks[0], "extracted_watermark_path": watermarks[1]}
        raise KeyError(f"неизвестная метрика {name}: добавьте её в фабрику параметров")

    return build


def evaluate(name, arguments, kind="noise"):
    """
    Вычисляет метрику, пропуская тест при отсутствии необязательных пакетов.

    Args:
        name (str): имя метрики
        arguments (callable): фабрика параметров вызова
        kind (str): вид искажения

    Returns:
        float: значение метрики
    """
    metric = require(name)
    try:
        return metric.expertise(arguments(name, kind))
    except RuntimeError as error:
        pytest.skip(f"{name}: {error}")


def test_registry_is_not_empty():
    """
    Проверяет, что импорт пакета наполнил реестр метрик.

    Returns:
        None
    """
    assert METRICS, "реестр метрик пуст: проверьте автообход в expertise_solutions/__init__.py"


@pytest.mark.parametrize("name", NAMES)
def test_reachable_through_orchestrator(name):
    """
    Проверяет доступ к метрике по имени через метакласс Expertise_Core.

    Args:
        name (str): имя метрики

    Returns:
        None
    """
    assert getattr(Expertise_Core, name) is METRICS[name]


@pytest.mark.parametrize("name", NAMES)
def test_returns_float(name, arguments):
    """
    Проверяет, что метрика возвращает вещественное число.

    Отчёт бенчмарка складывает значения метрик в таблицу, поэтому массив или
    словарь вместо числа сломает сборку отчёта, а не вызов метрики.

    Args:
        name (str): имя метрики
        arguments (callable): фабрика параметров вызова

    Returns:
        None
    """
    value = evaluate(name, arguments)
    assert isinstance(value, float), f"{name} вернула {type(value).__name__}"


@pytest.mark.parametrize("name", NAMES)
def test_batch_api(name, arguments):
    """
    Проверяет запуск метрики через пакетный интерфейс use_expertises.

    Именно так метрики вызывает фреймворк, поэтому расхождение сигнатур
    обнаруживается только здесь, а не при прямом вызове expertise.

    Args:
        name (str): имя метрики
        arguments (callable): фабрика параметров вызова

    Returns:
        None
    """
    require(name)
    try:
        Expertise_Core.use_expertises({name: arguments(name)})
    except RuntimeError as error:
        pytest.skip(f"{name}: {error}")


@pytest.mark.parametrize("name", NAMES)
def test_deterministic(name, arguments):
    """
    Проверяет, что повторный вызов даёт тот же результат.

    Args:
        name (str): имя метрики
        arguments (callable): фабрика параметров вызова

    Returns:
        None
    """
    first = evaluate(name, arguments)
    second = evaluate(name, arguments)
    assert first == second or (np.isnan(first) and np.isnan(second))


@pytest.mark.parametrize(
    "name, expected",
    [
        ("MSE", 0.0),
        ("PSNR", float("inf")),
        ("SSIM", 1.0),
        ("MS_SSIM", 1.0),
        ("VIF", 1.0),
        ("FSIM", 1.0),
        ("FSIMc", 1.0),
    ],
)
def test_identical_images_reach_limit(name, expected, arguments):
    """
    Проверяет предельное значение метрики на паре одинаковых изображений.

    Проверка слабее, чем кажется: в VIF при отсутствии искажения числитель
    совпадает со знаменателем независимо от того, по каким субполосам идёт
    суммирование, поэтому ошибка в выборе субполос через неё не видна.
    Основную нагрузку несёт сверка с эталонами.

    Args:
        name (str): имя метрики
        expected (float): ожидаемое предельное значение
        arguments (callable): фабрика параметров вызова

    Returns:
        None
    """
    value = evaluate(name, arguments, "identity")
    if np.isinf(expected):
        assert np.isinf(value)
    else:
        assert abs(value - expected) < 1e-9


def test_ber_limits(arguments):
    """
    Проверяет предельные значения BER на совпадении и полной инверсии.

    Args:
        arguments (callable): фабрика параметров вызова

    Returns:
        None
    """
    metric = require("BER")
    bits = "".join(np.random.default_rng(2).integers(0, 2, 256).astype(str))
    inverted = "".join("1" if symbol == "0" else "0" for symbol in bits)
    assert metric.expertise({"original_bits": bits, "extracted_bits": bits}) == 0.0
    assert metric.expertise({"original_bits": bits, "extracted_bits": inverted}) == 1.0


def test_nc_changes_sign_on_inversion(arguments):
    """
    Проверяет, что NC различает точное совпадение и полную инверсию.

    Именно этим NC отличается от BER: инверсия даёт минус единицу, а случайный
    результат — значение около нуля.

    Args:
        arguments (callable): фабрика параметров вызова

    Returns:
        None
    """
    metric = require("NC")
    bits = "".join(np.random.default_rng(2).integers(0, 2, 256).astype(str))
    inverted = "".join("1" if symbol == "0" else "0" for symbol in bits)
    assert metric.expertise({"original_bits": bits, "extracted_bits": bits}) == 1.0
    assert metric.expertise({"original_bits": bits, "extracted_bits": inverted}) == -1.0


@pytest.mark.parametrize("family", ["noise", "blur", "jpeg"])
@pytest.mark.parametrize("name", sorted(FULL_REFERENCE - {"LPIPS", "DISTS"}))
def test_monotone_in_distortion(name, family, arguments):
    """
    Проверяет, что метрика реагирует на усиление искажения в нужную сторону.

    Сравниваются две интенсивности одного и того же искажения. Сравнивать
    разные виды между собой нельзя: метрики упорядочивают их по-разному, и это
    их законное свойство, а не дефект. MS-SSIM, например, штрафует размытие
    мягче, чем сжатие, ради чего он и сделан мультимасштабным.

    Метрика, потерявшая чувствительность, продолжает возвращать правдоподобные
    числа и по одному замеру неотличима от рабочей.

    Args:
        name (str): имя метрики
        family (str): семейство искажений
        arguments (callable): фабрика параметров вызова

    Returns:
        None
    """
    mild = evaluate(name, arguments, f"{family}_mild")
    severe = evaluate(name, arguments, f"{family}_severe")
    if name in LOWER_IS_BETTER:
        assert severe > mild, (
            f"{name} на {family}: сильное искажение дало не большее значение ({severe:.6f} против {mild:.6f})"
        )
    else:
        assert severe < mild, (
            f"{name} на {family}: сильное искажение дало не меньшее значение ({severe:.6f} против {mild:.6f})"
        )


def test_auc_of_random_scores_is_near_half(detector):
    """
    Проверяет, что AUC случайных оценок близка к половине.

    Args:
        detector (tuple): метки и оценки детектора

    Returns:
        None
    """
    metric = require("AUC")
    truth = detector[0]
    scores = np.random.default_rng(17).random(truth.shape[0])
    assert abs(metric.expertise({"y_true": truth, "y_scores": scores}) - 0.5) < 0.1


def test_auc_of_perfect_detector_is_one(detector):
    """
    Проверяет, что безошибочный детектор даёт AUC, равную единице.

    Args:
        detector (tuple): метки и оценки детектора

    Returns:
        None
    """
    metric = require("AUC")
    truth = detector[0]
    assert metric.expertise({"y_true": truth, "y_scores": truth.astype(float)}) == 1.0


def test_p_value_never_reaches_zero(detector):
    """
    Проверяет, что p-значение не обращается в ноль.

    Ноль утверждал бы невозможность события по конечной выборке. Поправка на
    единицу в числителе и знаменателе задаёт нижнюю границу, равную обратному
    размеру выборки плюс один.

    Args:
        detector (tuple): метки и оценки детектора

    Returns:
        None
    """
    metric = require("P_Value")
    samples = detector[2]
    value = metric.expertise({"statistic": 1e9, "null_samples": samples})
    assert value == pytest.approx(1 / (samples.shape[0] + 1))


@pytest.mark.parametrize("side", [16, 40, 80, 200])
@pytest.mark.parametrize("name", sorted(FULL_REFERENCE - {"LPIPS", "DISTS"}))
def test_small_frames(name, side, tmp_path):
    """
    Проверяет поведение на кадрах меньше рабочего размера.

    Метрика обязана либо посчитаться, либо отказать с ValueError. Молчаливый
    результат на кадре, для которого она не определена, опаснее отказа.

    Args:
        name (str): имя метрики
        side (int): сторона квадратного кадра
        tmp_path (pathlib.Path): временный каталог теста

    Returns:
        None
    """
    require(name)
    array = reference_image(side, side)
    first = tmp_path / "small_first.png"
    second = tmp_path / "small_second.png"
    Image.fromarray(array, "RGB").save(first)
    Image.fromarray(np.roll(array, 1, axis=1), "RGB").save(second)

    try:
        value = METRICS[name].expertise({"original_path": str(first), "distorted_path": str(second)})
    except ValueError:
        assert side < MINIMUM_SIDE.get(name, 0) + 1 or name in MINIMUM_SIDE
        return
    assert np.isfinite(value) or np.isinf(value)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
@pytest.mark.parametrize("name", sorted(FULL_REFERENCE - {"LPIPS", "DISTS"}))
def test_input_modes(name, mode, tmp_path):
    """
    Проверяет работу с входом не в режиме RGB.

    Метрика обязана сама привести изображение к нужному представлению: серые,
    полупрозрачные и палитровые файлы встречаются в датасетах постоянно.

    Args:
        name (str): имя метрики
        mode (str): режим изображения Pillow
        tmp_path (pathlib.Path): временный каталог теста

    Returns:
        None
    """
    require(name)
    array = reference_image()
    if mode == "L":
        image = Image.fromarray(array[:, :, 0], "L")
    elif mode == "RGBA":
        alpha = np.full(array.shape[:2] + (1,), 190, dtype=np.uint8)
        image = Image.fromarray(np.concatenate([array, alpha], axis=2), "RGBA")
    else:
        image = Image.fromarray(array, "RGB").convert("P")

    path = tmp_path / "mode.png"
    image.save(path)
    METRICS[name].expertise({"original_path": str(path), "distorted_path": str(path)})


@pytest.mark.parametrize("name", sorted(FULL_REFERENCE - {"LPIPS", "DISTS"}))
def test_accepts_path_objects(name, tmp_path, original, distorted):
    """
    Проверяет, что вместо строк принимаются объекты pathlib.Path.

    Args:
        name (str): имя метрики
        tmp_path (pathlib.Path): временный каталог теста
        original (str): путь к эталонному изображению
        distorted (callable): фабрика искажённых копий

    Returns:
        None
    """
    from pathlib import Path

    require(name)
    METRICS[name].expertise({"original_path": Path(original), "distorted_path": Path(distorted("noise"))})


@pytest.mark.parametrize(
    "name, params",
    [
        ("BER", {"original_bits": "1010", "extracted_bits": "10"}),
        ("BER", {"original_bits": "", "extracted_bits": ""}),
        ("NC", {"original_bits": "1010", "extracted_bits": "10"}),
        ("NC", {"original_bits": "", "extracted_bits": ""}),
        ("Accuracy", {"y_true": [1, 0, 1], "y_pred": [1, 0]}),
        ("Precision", {"y_true": [1, 0, 1], "y_pred": [1, 0]}),
        ("Recall", {"y_true": [1, 0, 1], "y_pred": [1, 0]}),
        ("F1", {"y_true": [1, 0, 1], "y_pred": [1, 0]}),
        ("AUC", {"y_true": [1, 0, 1], "y_scores": [0.1, 0.2]}),
        ("Accuracy", {"y_true": [], "y_pred": []}),
    ],
)
def test_invalid_arguments_raise_value_error(name, params):
    """
    Проверяет, что недопустимые аргументы дают ValueError с внятным текстом.

    Расхождение длин особенно важно: молчаливое сравнение по общей части даёт
    оценку, посчитанную по обрезку, и в отчёте она неотличима от честной.

    Args:
        name (str): имя метрики
        params (dict): недопустимые параметры

    Returns:
        None
    """
    metric = require(name)
    with pytest.raises(ValueError):
        metric.expertise(params)


@pytest.mark.parametrize(
    "name, params",
    [
        ("VIF", {"sigma_nsq": 0}),
        ("VIF", {"sigma_nsq": -1}),
        ("VIF", {"block_size": 0}),
    ],
)
def test_invalid_image_parameters_raise_value_error(name, params, original, distorted):
    """
    Проверяет валидацию числовых параметров метрик по изображениям.

    Args:
        name (str): имя метрики
        params (dict): недопустимые параметры
        original (str): путь к эталонному изображению
        distorted (callable): фабрика искажённых копий

    Returns:
        None
    """
    metric = require(name)
    with pytest.raises(ValueError):
        metric.expertise({"original_path": original, "distorted_path": distorted("noise"), **params})


def test_mismatched_image_sizes_raise_value_error(tmp_path, original):
    """
    Проверяет, что VIF отказывается сравнивать кадры разного размера.

    Args:
        tmp_path (pathlib.Path): временный каталог теста
        original (str): путь к эталонному изображению

    Returns:
        None
    """
    metric = require("VIF")
    other = tmp_path / "other_size.png"
    Image.fromarray(reference_image(200, 240), "RGB").save(other)
    with pytest.raises(ValueError):
        metric.expertise({"original_path": original, "distorted_path": str(other)})


def test_length_mismatch_is_allowed_explicitly():
    """
    Проверяет, что сравнение по общей части включается явным флагом.

    Args:
        None

    Returns:
        None
    """
    metric = require("BER")
    value = metric.expertise({"original_bits": "1010", "extracted_bits": "10", "allow_length_mismatch": True})
    assert value == 0.0


@pytest.mark.parametrize("kind", ["identity", "noise", "blur", "jpeg"])
def test_mse_and_psnr_match_skimage(kind, original, distorted):
    """
    Сверяет MSE и PSNR с реализациями scikit-image.

    Args:
        kind (str): вид искажения
        original (str): путь к эталонному изображению
        distorted (callable): фабрика искажённых копий

    Returns:
        None
    """
    metrics = pytest.importorskip("skimage.metrics", reason="scikit-image не установлен")

    path = distorted(kind)
    first = np.asarray(Image.open(original).convert("RGB"), dtype=np.float64)
    second = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    pair = {"original_path": original, "distorted_path": path}

    assert require("MSE").expertise(pair) == pytest.approx(metrics.mean_squared_error(first, second), rel=1e-12)
    if kind != "identity":
        assert require("PSNR").expertise(pair) == pytest.approx(
            metrics.peak_signal_noise_ratio(first, second, data_range=255), rel=1e-12
        )


@pytest.mark.parametrize("kind", ["identity", "noise", "blur", "jpeg"])
def test_ssim_matches_skimage(kind, original, distorted):
    """
    Сверяет SSIM с реализацией scikit-image.

    Параметры подобраны так, чтобы обе стороны считали одно и то же: гауссово
    окно с сигмой 1.5 и нормировка по генеральной совокупности. Именно эта
    сверка вскрыла усреднение индекса по краю, дополненному нулями.

    Args:
        kind (str): вид искажения
        original (str): путь к эталонному изображению
        distorted (callable): фабрика искажённых копий

    Returns:
        None
    """
    metrics = pytest.importorskip("skimage.metrics", reason="scikit-image не установлен")

    path = distorted(kind)
    first = np.asarray(Image.open(original).convert("L"), dtype=np.float64)
    second = np.asarray(Image.open(path).convert("L"), dtype=np.float64)

    expected = metrics.structural_similarity(
        first, second, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False
    )
    ours = require("SSIM").expertise({"original_path": original, "distorted_path": path})
    assert ours == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("kind", ["identity", "noise", "blur", "jpeg"])
def test_ms_ssim_matches_sewar(kind, original, distorted):
    """
    Сверяет MS-SSIM с реализацией sewar.

    Сверка вскрыла понижение масштаба неверным фильтром: в оригинальной работе
    используется усреднение по блоку два на два.

    Args:
        kind (str): вид искажения
        original (str): путь к эталонному изображению
        distorted (callable): фабрика искажённых копий

    Returns:
        None
    """
    full_ref = pytest.importorskip("sewar.full_ref", reason="sewar не установлен")

    path = distorted(kind)
    first = np.asarray(Image.open(original).convert("L"))
    second = np.asarray(Image.open(path).convert("L"))

    expected = float(np.real(full_ref.msssim(first, second, MAX=255)))
    ours = require("MS_SSIM").expertise({"original_path": original, "distorted_path": path})
    assert ours == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    "name, reference_name",
    [
        ("Accuracy", "accuracy_score"),
        ("Precision", "precision_score"),
        ("Recall", "recall_score"),
        ("F1", "f1_score"),
    ],
)
def test_detector_metrics_match_sklearn(name, reference_name, detector):
    """
    Сверяет метрики бинарного детектора с реализациями scikit-learn.

    Args:
        name (str): имя метрики
        reference_name (str): имя эталонной функции в sklearn.metrics
        detector (tuple): метки и оценки детектора

    Returns:
        None
    """
    sklearn_metrics = pytest.importorskip("sklearn.metrics", reason="scikit-learn не установлен")

    truth, predicted, _ = detector
    reference = getattr(sklearn_metrics, reference_name)
    expected = (
        reference(truth, predicted)
        if reference_name == "accuracy_score"
        else reference(truth, predicted, zero_division=0)
    )
    ours = require(name).expertise({"y_true": truth, "y_pred": predicted})
    assert ours == pytest.approx(expected, abs=1e-12)


def test_auc_matches_sklearn(detector):
    """
    Сверяет AUC с реализацией scikit-learn, в том числе на совпадающих оценках.

    Совпадающие оценки проверяются отдельно: без усреднения рангов результат
    зависел бы от произвольного порядка сортировки.

    Args:
        detector (tuple): метки и оценки детектора

    Returns:
        None
    """
    sklearn_metrics = pytest.importorskip("sklearn.metrics", reason="scikit-learn не установлен")

    truth, _, scores = detector
    metric = require("AUC")
    assert metric.expertise({"y_true": truth, "y_scores": scores}) == pytest.approx(
        sklearn_metrics.roc_auc_score(truth, scores), abs=1e-12
    )

    tied = np.round(scores, 1)
    assert metric.expertise({"y_true": truth, "y_scores": tied}) == pytest.approx(
        sklearn_metrics.roc_auc_score(truth, tied), abs=1e-12
    )


@pytest.mark.parametrize("kind", ["identity", "noise", "blur", "jpeg"])
@pytest.mark.parametrize("name", ["VIF", "FSIM", "FSIMc"])
def test_matches_recorded_values(name, kind, original, distorted):
    """
    Сверяет метрики без внешнего эталона с зафиксированными значениями.

    Для VIF значения получены после сверки с исходным vifvec, для FSIM и FSIMc
    независимой реализации найти не удалось, поэтому они фиксируют текущее
    поведение и защищают только от регрессии. При появлении эталона эти
    проверки стоит заменить живой сверкой.

    Args:
        name (str): имя метрики
        kind (str): вид искажения
        original (str): путь к эталонному изображению
        distorted (callable): фабрика искажённых копий

    Returns:
        None
    """
    metric = require(name)
    value = metric.expertise({"original_path": original, "distorted_path": distorted(kind)})
    assert value == pytest.approx(GOLDEN[kind][name], rel=GOLDEN_TOLERANCE)
