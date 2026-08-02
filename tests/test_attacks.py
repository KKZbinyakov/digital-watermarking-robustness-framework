"""Тесты корректности готовых атак.

Набор самонаполняющийся: атаки берутся из реестра Attack_Core, поэтому новый
класс атаки попадает под общие проверки автоматически, без правки этого файла.

Тесты разбиты на четыре группы:

    контракт      атака зарегистрирована, доступна через оркестратор, пишет файл
                  и не портит вход
    семантика     атака с параметрами по умолчанию действительно меняет
                  изображение, а атака без потерь — не меняет
    устойчивость  вырожденные размеры и нестандартные режимы входа
    инварианты    свойства конкретных алгоритмов, нарушение которых означает
                  тихую неправильность результата, а не падение

Последняя группа самая важная: ошибки, найденные при разработке этих атак,
падений не вызывали. Чтение за концом массива в CLAHE возвращало воспроизводимо
неправильную картинку, а атака яркости с дефолтными параметрами не делала ничего
и давала нулевой BER.
"""

import hashlib
import inspect

import numpy as np
import pytest
from PIL import Image

import dwarf
import dwarf.ready_solutions.attack_solutions  # noqa: F401  наполняет реестр атак
from dwarf import Attack_Core

LOSSLESS = frozenset({"Tiff", "Flif"})
"""Атаки без потерь: пиксели обязаны остаться прежними, BER на них нулевой."""

CHANGES_SHAPE = frozenset({"Crop"})
"""Атаки, меняющие размер кадра: для них проверка совпадения формы неприменима."""

SEEDED = frozenset({"Color_Jitter", "Color_Space_Noise", "Color_Quantization"})
"""Атаки со случайностью: обязаны быть воспроизводимы по seed и различаться без него."""

FRAGILE_ON_TINY = frozenset({"Crop"})
"""Атаки, не переживающие вырожденные размеры. Известный дефект, см. test_degenerate_sizes."""


def _discover():
    """
    Собирает конкретные классы атак из реестра Attack_Core.

    Классы-категории (Ready_Geometric_Attacks и прочие) не реализуют attack и
    потому абстрактны — по этому признаку они и отсеиваются.

    Returns:
        dict: отображение имени атаки в её класс
    """
    return {name: cls for name, cls in Attack_Core.get_registered_attacks().items() if not inspect.isabstract(cls)}


ATTACKS = _discover()
NAMES = sorted(ATTACKS)


def run(name, tmp_path, source, **params):
    """
    Запускает атаку и возвращает результат массивом.

    Атаки, которым нужны отсутствующие кодеки или внешние утилиты, поднимают
    RuntimeError с подсказкой по установке. Такой тест помечается пропущенным,
    а не проваленным: отсутствие необязательной зависимости не является дефектом.

    Args:
        name (str): имя атаки в реестре
        tmp_path (pathlib.Path): временный каталог теста
        source (str): путь к исходному изображению
        params: дополнительные параметры атаки

    Returns:
        np.ndarray: результат атаки, массив (H, W, 3) типа uint8
    """
    output = tmp_path / f"{name}_out.png"
    try:
        ATTACKS[name].attack({"input_data": str(source), "output_data": str(output), **params})
    except RuntimeError as error:
        pytest.skip(f"{name}: {error}")
    assert output.exists(), f"{name} не создала выходной файл"
    return np.asarray(Image.open(output).convert("RGB"))


@pytest.fixture
def photo(tmp_path):
    """
    Готовит тестовое изображение с градиентами и мелкой текстурой.

    Плавные градиенты нужны, чтобы были видны артефакты квантования и дизеринга,
    шумовой канал — чтобы гистограмма не вырождалась, а сплошные пятна — чтобы
    проверялось поведение на однородных областях.

    Args:
        tmp_path (pathlib.Path): временный каталог теста

    Returns:
        str: путь к сохранённому изображению
    """
    height, width = 96, 128
    rng = np.random.default_rng(20240501)
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    array[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    array[:, :, 2] = (rng.random((height, width)) * 255).astype(np.uint8)
    array[10:30, 10:30] = 64
    array[50:70, 80:110] = 200

    path = tmp_path / "photo.png"
    Image.fromarray(array, "RGB").save(path)
    return str(path)


@pytest.fixture
def photolike():
    """
    Возвращает фабрику фотоподобных изображений с умеренной цветностью.

    Основная фикстура photo намеренно недружелюбна: два канала заняты
    полноразмерными градиентами, третий шумом, поэтому цветность экстремальна и
    обратный перевод YCbCr в RGB упирается в границы диапазона примерно на
    четверти пикселей. Для сверки с эталоном это мешает: измеряется поведение
    цветового конвейера, а не самого алгоритма.

    Returns:
        callable: функция (высота, ширина) -> np.ndarray формы (H, W, 3) типа uint8
    """

    def build(height=96, width=128, seed=20240501):
        rng = np.random.default_rng(seed)
        rows, columns = np.mgrid[0:height, 0:width].astype(float)
        base = 0.5 + 0.25 * np.sin(2 * np.pi * columns / width) + 0.15 * np.cos(2 * np.pi * rows / height)
        array = np.stack([base * 210 + 20, base * 190 + 30, base * 170 + 40], axis=-1)
        array[height // 5 : height // 2, width // 5 : width // 2] *= 0.55
        array[3 * height // 5 :, 3 * width // 5 :] *= 1.5
        array += rng.normal(0, 6, array.shape)
        return np.clip(array, 0, 255).astype(np.uint8)

    return build


# ----------------------------------------------------------------------------
# Контракт
# ----------------------------------------------------------------------------


def test_registry_is_not_empty():
    """
    Проверяет, что импорт пакета наполнил реестр атак.

    Регистрация идёт через __init_subclass__, то есть побочным эффектом импорта
    модуля. Пустой реестр означает, что автообход в attack_solutions/__init__.py
    сломан и Attack_Core.<Имя> перестал работать.

    Returns:
        None
    """
    assert ATTACKS, "реестр атак пуст: проверьте автообход в attack_solutions/__init__.py"


@pytest.mark.parametrize("name", NAMES)
def test_reachable_through_orchestrator(name):
    """
    Проверяет доступ к атаке по имени через метакласс Attack_Core.

    Args:
        name (str): имя атаки

    Returns:
        None
    """
    assert getattr(Attack_Core, name) is ATTACKS[name]


@pytest.mark.parametrize("name", NAMES)
def test_batch_api(name, tmp_path, photo):
    """
    Проверяет запуск атаки через пакетный интерфейс use_attacks.

    Именно так атаки вызывает фреймворк, поэтому расхождение сигнатур
    обнаруживается только здесь, а не при прямом вызове attack.

    Args:
        name (str): имя атаки
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    output = tmp_path / "batch.png"
    try:
        Attack_Core.use_attacks({name: {"input_data": photo, "output_data": str(output)}})
    except RuntimeError as error:
        pytest.skip(f"{name}: {error}")
    assert output.exists()


@pytest.mark.parametrize("name", NAMES)
def test_input_file_untouched(name, tmp_path, photo):
    """
    Проверяет, что атака не изменяет исходный файл.

    Args:
        name (str): имя атаки
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    before = hashlib.sha256(open(photo, "rb").read()).hexdigest()
    run(name, tmp_path, photo)
    after = hashlib.sha256(open(photo, "rb").read()).hexdigest()
    assert before == after, f"{name} изменила исходный файл"


@pytest.mark.parametrize("name", sorted(set(NAMES) - CHANGES_SHAPE))
def test_shape_preserved(name, tmp_path, photo):
    """
    Проверяет, что размер кадра не изменился.

    Args:
        name (str): имя атаки
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    expected = np.asarray(Image.open(photo).convert("RGB")).shape
    assert run(name, tmp_path, photo).shape == expected


# ----------------------------------------------------------------------------
# Семантика
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(set(NAMES) - LOSSLESS - CHANGES_SHAPE))
def test_defaults_actually_attack(name, tmp_path, photo):
    """
    Проверяет, что атака с параметрами по умолчанию меняет изображение.

    Атака, которая с дефолтами возвращает вход без изменений, в прогоне
    бенчмарка тихо даёт нулевой BER и создаёт ложное впечатление стойкости ЦВЗ.
    Так вело себя изменение яркости с множителями 1.0.

    Args:
        name (str): имя атаки
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    source = np.asarray(Image.open(photo).convert("RGB"))
    result = run(name, tmp_path, photo)
    assert not np.array_equal(result, source), (
        f"{name} с параметрами по умолчанию не изменила изображение: в бенчмарке это даст ложный нулевой BER"
    )


@pytest.mark.parametrize("name", sorted(LOSSLESS))
def test_lossless_is_bit_exact(name, tmp_path, photo):
    """
    Проверяет, что атака без потерь оставляет пиксели неизменными.

    Такие атаки контрольные: ненулевой BER на них означает ошибку в схеме
    встраивания или скрытое пересжатие, а не стойкость к смене контейнера.

    Args:
        name (str): имя атаки
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    source = np.asarray(Image.open(photo).convert("RGB"))
    assert np.array_equal(run(name, tmp_path, photo), source)


@pytest.mark.parametrize("name", NAMES)
def test_deterministic(name, tmp_path, photo):
    """
    Проверяет воспроизводимость результата при одинаковом seed.

    Args:
        name (str): имя атаки
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    first = run(name, tmp_path, photo, seed=7)
    second = run(name, tmp_path, photo, seed=7)
    assert np.array_equal(first, second), f"{name} невоспроизводима при одинаковом seed"


@pytest.mark.parametrize("name", sorted(SEEDED))
def test_seed_changes_result(name, tmp_path, photo):
    """
    Проверяет, что seed действительно управляет случайностью.

    Если атака игнорирует seed, воспроизводимость из предыдущего теста может
    достигаться тем, что случайности нет вовсе.

    Args:
        name (str): имя атаки
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    extra = {"method": "kmeans"} if name == "Color_Quantization" else {}
    first = run(name, tmp_path, photo, seed=1, **extra)
    second = run(name, tmp_path, photo, seed=2, **extra)
    assert not np.array_equal(first, second), f"{name} игнорирует seed"


# ----------------------------------------------------------------------------
# Устойчивость
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 1), (1, 64), (64, 1), (3, 3), (5, 7)])
@pytest.mark.parametrize("name", NAMES)
def test_degenerate_sizes(name, shape, tmp_path):
    """
    Проверяет работу на вырожденных размерах кадра.

    Изображение меньше тайла CLAHE, матрицы Байера или блока кодека — типичный
    источник выхода за границы массива и деления на ноль.

    Args:
        name (str): имя атаки
        shape (tuple): размер тестового изображения (высота, ширина)
        tmp_path (pathlib.Path): временный каталог теста

    Returns:
        None
    """
    if name in FRAGILE_ON_TINY:
        pytest.xfail(f"{name}: известный дефект, срез нулевой ширины при малых размерах")

    height, width = shape
    rng = np.random.default_rng(0)
    path = tmp_path / "tiny.png"
    Image.fromarray((rng.random((height, width, 3)) * 255).astype(np.uint8), "RGB").save(path)
    run(name, tmp_path, path)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "I;16"])
@pytest.mark.parametrize("name", NAMES)
def test_input_modes(name, mode, tmp_path):
    """
    Проверяет работу с входом не в режиме RGB.

    Атака обязана сама привести изображение к RGB: серые, полупрозрачные и
    палитровые файлы встречаются в датасетах постоянно.

    Args:
        name (str): имя атаки
        mode (str): режим изображения Pillow
        tmp_path (pathlib.Path): временный каталог теста

    Returns:
        None
    """
    rng = np.random.default_rng(1)
    base = (rng.random((32, 40, 3)) * 255).astype(np.uint8)
    if mode == "L":
        image = Image.fromarray(base[:, :, 0], "L")
    elif mode == "RGBA":
        alpha = np.full((32, 40, 1), 180, dtype=np.uint8)
        image = Image.fromarray(np.concatenate([base, alpha], axis=2), "RGBA")
    elif mode == "P":
        image = Image.fromarray(base, "RGB").convert("P")
    else:
        image = Image.fromarray(base[:, :, 0].astype(np.uint16) * 257).convert("I;16")

    path = tmp_path / "mode.png"
    image.save(path)
    run(name, tmp_path, path)


# ----------------------------------------------------------------------------
# Инварианты алгоритмов
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("levels", [2, 4, 8])
def test_dithering_preserves_mean(levels, tmp_path, photo):
    """
    Проверяет определяющее свойство диффузии ошибки: сохранение средней яркости.

    Ошибка квантования не отбрасывается, а раздаётся соседям, поэтому среднее по
    кадру почти не меняется. Если вместо диффузии подставить обычное квантование
    или палитровый дизеринг Pillow, среднее уедет — а глазом подмену не заметить.

    Args:
        levels (int): число уровней квантования на канал
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    source = np.asarray(Image.open(photo).convert("RGB")).astype(float)
    result = run("Dithering", tmp_path, photo, levels=levels).astype(float)
    assert abs(result.mean() - source.mean()) < 1.0


@pytest.mark.parametrize("levels", [2, 3, 4, 8])
def test_dithering_level_count(levels, tmp_path, photo):
    """
    Проверяет, что дизеринг оставляет ровно запрошенное число уровней.

    Args:
        levels (int): число уровней квантования на канал
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    result = run("Dithering", tmp_path, photo, levels=levels)
    assert len(np.unique(result)) <= levels


@pytest.mark.parametrize("colors", [2, 8, 16, 64])
def test_quantization_palette_size(colors, tmp_path, photo):
    """
    Проверяет, что палитра содержит ровно запрошенное число цветов.

    Тихое усечение палитры при colors больше размера выборки давало меньше
    цветов, чем просили, и это никак не проявлялось в выводе.

    Args:
        colors (int): запрошенное число цветов
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    result = run("Color_Quantization", tmp_path, photo, method="kmeans", colors=colors, seed=0)
    unique = np.unique(result.reshape(-1, 3), axis=0)
    assert len(unique) <= colors


@pytest.mark.parametrize("bins", [128, 64, 16])
def test_clahe_bins_preserve_structure(bins, tmp_path, photo):
    """
    Регрессионный тест на чтение за концом массива в CLAHE.

    Отображения тайлов имеют форму (tiles, tiles, bins), а индексировались сырым
    значением пикселя 0..255. При bins меньше 256 индекс уезжал за пределы
    массива: с выключенной проверкой границ это не исключение, а чтение чужой
    памяти, дающее воспроизводимо неправильный, но правдоподобный результат.

    Сравнивается структура, а не яркость. Порог обрезания гистограммы равен
    clip_limit * площадь тайла / bins, то есть напрямую зависит от числа корзин,
    поэтому среднее по кадру при разных bins обязано различаться. А вот
    пространственная структура сохраняется: измеренная корреляция не опускается
    ниже 0.94 вплоть до 16 корзин, тогда как чтение чужой памяти её разрушает.

    Args:
        bins (int): число корзин гистограммы
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    reference = run("Histogram_Equalization", tmp_path, photo, method="clahe", bins=256).astype(float)
    result = run("Histogram_Equalization", tmp_path, photo, method="clahe", bins=bins).astype(float)
    correlation = np.corrcoef(result.ravel(), reference.ravel())[0, 1]
    assert correlation > 0.9, f"при bins={bins} корреляция с bins=256 всего {correlation:.4f}"


@pytest.mark.parametrize("height, tiles", [(100, 8), (97, 8), (33, 4), (255, 16)])
def test_clahe_tiles_cover_whole_frame(height, tiles):
    """
    Проверяет, что тайлы CLAHE покрывают кадр целиком.

    Размер тайла округляется вверх. При округлении вниз правый и нижний края не
    попадают ни в одну гистограмму: результат остаётся правдоподобным, поэтому
    через выходное изображение дефект не виден, и проверять приходится сам
    расчёт разбиения.

    Так уже ошибались: выражение с np.ceil под директивой cdivision считалось
    целочисленным делением и работало как floor.

    Args:
        height (int): высота кадра, не кратная числу тайлов
        tiles (int): число тайлов по оси

    Returns:
        None
    """
    from dwarf.ready_solutions.attack_solutions.color_brightness.histogram_equalization import _build_maps

    channel = np.zeros((height, height), dtype=np.uint8)
    _, tile_height, tile_width = _build_maps(channel, tiles, 2.0, 256)

    assert tile_height * tiles >= height, (
        f"тайлы покрывают {tile_height * tiles:.0f} строк из {height}: "
        f"нижний край кадра не попадает ни в одну гистограмму"
    )
    assert tile_width * tiles >= height
    assert (tile_height - 1) * tiles < height, "тайлы неоправданно крупные"


def test_save_rgb_rounds_not_truncates(tmp_path):
    """
    Проверяет, что сохранение округляет дробные значения, а не отбрасывает их.

    Приведение к uint8 усекает: 127.6 превращается в 127. Это систематическое
    смещение на уровень вниз по каждому пикселю, то есть ровно в том разряде,
    в котором живёт ЦВЗ. Дробные значения возникают в дизеринге при нечётном
    числе уровней и в центрах палитры k-means.

    Args:
        tmp_path (pathlib.Path): временный каталог теста

    Returns:
        None
    """
    from dwarf.ready_solutions.utils.attack_utils import save_rgb

    path = tmp_path / "rounded.png"
    save_rgb(np.full((4, 4, 3), 127.6), str(path))
    assert np.asarray(Image.open(path))[0, 0, 0] == 128

    save_rgb(np.full((4, 4, 3), 127.4), str(path))
    assert np.asarray(Image.open(path))[0, 0, 0] == 127


def test_dithering_uses_fractional_levels(tmp_path, photo):
    """
    Проверяет корректность уровней при нечётном числе градаций.

    При levels равном 3 шаг квантования равен 127.5, то есть уровни дробные.
    Усечение вместо округления сместило бы средний уровень со 128 на 127.

    Args:
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    result = run("Dithering", tmp_path, photo, levels=3)
    assert set(np.unique(result)) <= {0, 128, 255}


def test_grayscale_channels_are_equal(tmp_path, photo):
    """
    Проверяет, что после обесцвечивания все три канала совпадают.

    Args:
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    result = run("Grayscale", tmp_path, photo)
    assert np.array_equal(result[:, :, 0], result[:, :, 1])
    assert np.array_equal(result[:, :, 1], result[:, :, 2])


def test_gamma_one_is_identity(tmp_path, photo):
    """
    Проверяет, что гамма, равная единице, не меняет изображение.

    Отклонение здесь означает ошибку округления в таблице подстановки: усечение
    вместо округления смещает каждый пиксель на уровень вниз, то есть ровно в том
    разряде, в котором живёт ЦВЗ.

    Args:
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    source = np.asarray(Image.open(photo).convert("RGB"))
    assert np.array_equal(run("Gamma_Correction", tmp_path, photo, gamma=1.0), source)


def test_bit_depth_eight_is_identity(tmp_path, photo):
    """
    Проверяет, что сохранение всех восьми бит не меняет изображение.

    Args:
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    source = np.asarray(Image.open(photo).convert("RGB"))
    assert np.array_equal(run("Bit_Depth_Reduction", tmp_path, photo, bits=8), source)


@pytest.mark.parametrize("bits, expected", [(1, 2), (2, 4), (4, 16)])
def test_bit_depth_level_count(bits, expected, tmp_path, photo):
    """
    Проверяет, что после огрубления остаётся ровно 2 в степени bits уровней.

    Args:
        bits (int): сколько старших бит оставлено
        expected (int): ожидаемое число различных уровней
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    result = run("Bit_Depth_Reduction", tmp_path, photo, bits=bits)
    assert len(np.unique(result)) <= expected


@pytest.mark.parametrize(
    "name, low, high",
    [
        ("Jpeg", "quality", "quality"),
        ("Webp", "quality", "quality"),
    ],
)
def test_compression_quality_is_monotone(name, low, high, tmp_path, photo):
    """
    Проверяет, что более высокое качество кодека даёт меньшее искажение.

    Нарушение монотонности означает, что параметр качества не доходит до кодека
    или интерпретируется наоборот.

    Args:
        name (str): имя атаки
        low (str): имя параметра качества
        high (str): имя параметра качества
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    source = np.asarray(Image.open(photo).convert("RGB")).astype(float)
    worse = np.abs(run(name, tmp_path, photo, **{low: 10}).astype(float) - source).mean()
    better = np.abs(run(name, tmp_path, photo, **{high: 95}).astype(float) - source).mean()
    assert better < worse


# ----------------------------------------------------------------------------
# Валидация параметров
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, params",
    [
        ("Dithering", {"levels": 1}),
        ("Dithering", {"method": "нет такого"}),
        ("Dithering", {"method": "ordered", "matrix_size": 1}),
        ("Histogram_Equalization", {"method": "нет такого"}),
        ("Histogram_Equalization", {"method": "clahe", "tiles": 1}),
        ("Histogram_Equalization", {"method": "clahe", "bins": 1}),
        ("Histogram_Equalization", {"method": "clahe", "bins": 512}),
        ("Color_Quantization", {"colors": 1}),
        ("Color_Quantization", {"method": "нет такого"}),
        ("Color_Quantization", {"method": "kmeans", "colors": 64, "sample_size": 32}),
        ("Gamma_Correction", {"gamma": 0}),
        ("Gamma_Correction", {"gamma": -1}),
        ("Bit_Depth_Reduction", {"bits": 0}),
        ("Bit_Depth_Reduction", {"bits": 9}),
    ],
)
def test_invalid_parameters_raise_value_error(name, params, tmp_path, photo):
    """
    Проверяет, что недопустимые параметры дают ValueError с внятным текстом.

    Без проверки такие значения уходят вглубь numpy и падают делением на ноль
    или выходом за границы — по сообщению об ошибке причину не восстановить.

    Args:
        name (str): имя атаки
        params (dict): недопустимые параметры
        tmp_path (pathlib.Path): временный каталог теста
        photo (str): путь к тестовому изображению

    Returns:
        None
    """
    with pytest.raises(ValueError):
        ATTACKS[name].attack(
            {
                "input_data": photo,
                "output_data": str(tmp_path / "invalid.png"),
                **params,
            }
        )


# ----------------------------------------------------------------------------
# Сверка с эталонными реализациями
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(96, 128), (240, 320)])
def test_clahe_matches_opencv(size, tmp_path, photolike):
    """
    Сверяет CLAHE с реализацией OpenCV на кадрах разного размера.

    Мелкие кадры проверяются отдельно не для галочки: при малой площади тайла
    порог обрезания опускается до одного отсчёта на корзину, и остаток срезанной
    массы составляет заметную долю тайла. Реализация, которая этот остаток
    теряет, на крупных кадрах совпадает с эталоном, а на мелких расходится —
    ровно так и был найден соответствующий дефект.

    Точного совпадения не ожидается: OpenCV работает прямо с каналом яркости,
    а атака переводит результат обратно в RGB с обрезанием по диапазону. Порог
    выставлен по измеренным значениям 0.989 и 0.999; потеря остатка срезанной
    массы, ради поимки которой тест и написан, роняла корреляцию до 0.69.

    Args:
        size (tuple): размер тестового кадра (высота, ширина)
        tmp_path (pathlib.Path): временный каталог теста
        photolike (callable): фабрика фотоподобных изображений

    Returns:
        None
    """
    cv2 = pytest.importorskip("cv2", reason="OpenCV не установлен, сверка пропущена")

    height, width = size
    source = photolike(height, width)
    path = tmp_path / "reference.png"
    Image.fromarray(source, "RGB").save(path)

    output = tmp_path / "reference_out.png"
    ATTACKS["Histogram_Equalization"].attack(
        {
            "input_data": str(path),
            "output_data": str(output),
            "method": "clahe",
            "tiles": 8,
            "clip_limit": 2.0,
        }
    )

    mine = np.asarray(Image.open(output).convert("YCbCr"))[:, :, 0].astype(float)
    luma = cv2.cvtColor(source, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    reference = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luma).astype(float)

    correlation = np.corrcoef(mine.ravel(), reference.ravel())[0, 1]
    assert correlation > 0.98, f"корреляция с OpenCV всего {correlation:.4f} на кадре {height}x{width}"
