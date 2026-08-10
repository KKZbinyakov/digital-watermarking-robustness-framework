"""Тесты атак.

Проверяются две вещи: общий контракт, одинаковый для всех атак, и частные
свойства отдельных атак. Общая часть параметризуется прямо из реестра, поэтому
новая атака попадает под проверки, ничего не дописывая сюда.
"""

import numpy as np
import pytest
from conftest import make_photo, solutions

import dwarf.ready_solutions.attack_solutions  # noqa: F401  наполняет реестр атак
from dwarf.core.attack_orchestrator.attack_core import Attack_Core

ATTACKS = solutions(Attack_Core.get_registered_attacks())
NAMES = sorted(ATTACKS)

LOSSLESS = frozenset({"Tiff", "Flif"})
"""Атаки без потерь: обязаны возвращать матрицу, совпадающую с входом побитово."""

SEEDED = frozenset({"Color_Jitter", "Color_Space_Noise"})
"""Атаки, у которых seed влияет на результат при значениях по умолчанию."""


def run(name, image, **params):
    """
    Вызывает атаку, пропуская тест при отсутствии внешнего кодека.

    Args:
        name (str): имя атаки в реестре
        image (np.ndarray): матрица изображения
        params: параметры атаки

    Returns:
        np.ndarray: матрица изображения после атаки
    """
    try:
        return Attack_Core.get_attack_class_by_name(name).attack(input_image=image, **params)
    except RuntimeError as error:
        pytest.skip(f"{name}: {error}")


# --- общий контракт -------------------------------------------------------


def test_registry_is_not_empty():
    assert ATTACKS, "реестр атак пуст: категории не импортировались"


@pytest.mark.parametrize("name", NAMES)
def test_reachable_through_orchestrator(name):
    assert Attack_Core.get_attack_class_by_name(name) is getattr(Attack_Core, name)


@pytest.mark.parametrize("name", NAMES)
def test_returns_image_matrix(name, small_photo):
    result = run(name, small_photo)
    assert isinstance(result, np.ndarray), f"{name} вернула {type(result).__name__}, а не матрицу"
    assert result.dtype == np.uint8, f"{name} вернула {result.dtype}, ожидается uint8"
    assert result.ndim == 3 and result.shape[2] == 3, f"{name} вернула форму {result.shape}"


@pytest.mark.parametrize("name", NAMES)
def test_result_is_writable(name, small_photo):
    assert run(name, small_photo).flags.writeable, f"{name} вернула неизменяемую матрицу"


@pytest.mark.parametrize("name", NAMES)
def test_input_matrix_untouched(name, small_photo):
    before = small_photo.copy()
    run(name, small_photo)
    assert np.array_equal(small_photo, before), f"{name} изменила входную матрицу на месте"


@pytest.mark.parametrize("name", NAMES)
def test_defaults_are_self_sufficient(name, small_photo):
    """Правило 2: атака обязана работать, получив только изображение."""
    run(name, small_photo)


@pytest.mark.parametrize("name", NAMES)
def test_shape_preserved_with_defaults(name, small_photo):
    assert run(name, small_photo).shape == small_photo.shape


@pytest.mark.parametrize("name", NAMES)
def test_output_feeds_next_attack(name, small_photo):
    """Результат атаки обязан годиться на вход другой: цепочки должны собираться."""
    assert run("Jpeg", run(name, small_photo)).shape == small_photo.shape


@pytest.mark.parametrize("name", NAMES)
def test_deterministic(name, small_photo):
    if name in SEEDED:
        pytest.skip(f"{name} случайна без явного seed")
    assert np.array_equal(run(name, small_photo), run(name, small_photo))


@pytest.mark.parametrize("name", sorted(SEEDED))
def test_seed_makes_reproducible(name, small_photo):
    assert np.array_equal(run(name, small_photo, seed=11), run(name, small_photo, seed=11))


@pytest.mark.parametrize("name", sorted(SEEDED))
def test_different_seeds_give_different_results(name, small_photo):
    assert not np.array_equal(run(name, small_photo, seed=1), run(name, small_photo, seed=2))


@pytest.mark.parametrize("name", NAMES)
def test_rejects_non_rgb_matrix(name):
    """Атаки на внешних кодеках проверяют зависимость раньше входа, поэтому их пропускаем."""
    with pytest.raises(ValueError):
        run(name, np.zeros((8, 8)))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "convert",
    [lambda a: a.astype(np.float64), lambda a: a.tolist()],
    ids=["float64", "list"],
)
def test_accepts_other_input_types(name, convert, small_photo):
    assert run(name, convert(small_photo)).shape == small_photo.shape


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("shape", [(1, 1), (1, 16), (16, 1), (3, 3)])
def test_degenerate_sizes(name, shape):
    """Кадр в несколько пикселей не должен ронять атаку."""
    result = run(name, make_photo(shape[0], shape[1], seed=3))
    assert result.ndim == 3 and result.dtype == np.uint8


@pytest.mark.parametrize("name", sorted(LOSSLESS))
def test_lossless_is_bit_exact(name, photo):
    assert np.array_equal(run(name, photo), photo), f"{name} заявлена без потерь, но изменила пиксели"


@pytest.mark.parametrize("name", sorted(set(NAMES) - LOSSLESS))
def test_defaults_actually_distort(name, photo):
    """Атака со значениями по умолчанию обязана что-то менять, иначе она бесполезна."""
    assert not np.array_equal(run(name, photo), photo), f"{name} со значениями по умолчанию ничего не изменила"


# --- сжатие ---------------------------------------------------------------


def test_webp_lossless_is_bit_exact(photo):
    assert np.array_equal(run("Webp", photo, lossless=True), photo)


@pytest.mark.parametrize("worse, better", [(5, 95), (30, 90), (50, 95)])
def test_jpeg_lower_quality_distorts_more(worse, better, photo):
    reference = photo.astype(np.float64)
    error_worse = np.abs(run("Jpeg", photo, quality=worse).astype(np.float64) - reference).mean()
    error_better = np.abs(run("Jpeg", photo, quality=better).astype(np.float64) - reference).mean()
    assert error_worse > error_better


def test_jpeg2000_higher_ratio_distorts_more(photo):
    reference = photo.astype(np.float64)
    mild = np.abs(run("Jpeg2000", photo, compression_ratio=5).astype(np.float64) - reference).mean()
    harsh = np.abs(run("Jpeg2000", photo, compression_ratio=200).astype(np.float64) - reference).mean()
    assert harsh > mild


# --- геометрия ------------------------------------------------------------


@pytest.mark.parametrize("mode", ["pad", "resize", "raw"])
def test_crop_modes_return_matrix(mode, photo):
    result = run("Crop", photo, mode=mode)
    assert result.dtype == np.uint8 and result.ndim == 3


@pytest.mark.parametrize("mode", ["pad", "resize"])
def test_crop_restores_original_size(mode, photo):
    assert run("Crop", photo, mode=mode).shape == photo.shape


@pytest.mark.parametrize("ratio, expected", [(0.5, (144, 144)), (0.25, (72, 72)), (1.0, (288, 288))])
def test_crop_raw_size_follows_ratio(ratio, expected, photo):
    assert run("Crop", photo, mode="raw", ratio=ratio).shape[:2] == expected


@pytest.mark.parametrize("mode", ["pad", "resize", "raw"])
def test_crop_ratio_one_is_identity(mode, photo):
    assert np.array_equal(run("Crop", photo, ratio=1.0, mode=mode), photo)


@pytest.mark.parametrize(
    "position, corner",
    [
        ("top_left", (0, 0)),
        ("top_right", (0, 144)),
        ("bottom_left", (144, 0)),
        ("bottom_right", (144, 144)),
        ("center", (72, 72)),
    ],
)
def test_crop_anchors_keep_expected_region(position, corner, photo):
    """Заливка контрастным цветом показывает, какая область осталась."""
    result = run("Crop", photo, position=position, fill=(255, 0, 255))
    rows, columns = np.where((result != [255, 0, 255]).any(axis=2))
    assert (rows.min(), columns.min()) == corner


def test_crop_random_position_is_reproducible(photo):
    assert np.array_equal(
        run("Crop", photo, position="random", seed=4),
        run("Crop", photo, position="random", seed=4),
    )


def test_crop_fill_colours_the_discarded_area(photo):
    result = run("Crop", photo, position="top_left", fill=(10, 20, 30))
    assert tuple(result[-1, -1]) == (10, 20, 30)


@pytest.mark.parametrize(
    "params",
    [
        {"ratio": 0},
        {"ratio": 1.5},
        {"position": "middle"},
        {"mode": "stretch"},
        {"mode": "resize", "resample": "spline"},
        {"fill": (1, 2)},
        {"fill": 300},
    ],
)
def test_crop_invalid_parameters_raise(params, small_photo):
    with pytest.raises(ValueError):
        run("Crop", small_photo, **params)


# --- цвет и яркость -------------------------------------------------------


def test_grayscale_channels_are_equal(photo):
    result = run("Grayscale", photo)
    assert np.array_equal(result[:, :, 0], result[:, :, 1])
    assert np.array_equal(result[:, :, 1], result[:, :, 2])


def test_gamma_one_is_identity(photo):
    assert np.array_equal(run("Gamma_Correction", photo, gamma=1.0), photo)


@pytest.mark.parametrize("gamma, brighter", [(0.5, True), (2.0, False)])
def test_gamma_direction(gamma, brighter, photo):
    result = run("Gamma_Correction", photo, gamma=gamma).astype(np.float64)
    assert bool(result.mean() > photo.astype(np.float64).mean()) is brighter


def test_bit_depth_eight_is_identity(photo):
    assert np.array_equal(run("Bit_Depth_Reduction", photo, bits=8), photo)


@pytest.mark.parametrize("bits, expected", [(1, 2), (2, 4), (3, 8), (4, 16)])
def test_bit_depth_level_count(bits, expected, photo):
    assert len(np.unique(run("Bit_Depth_Reduction", photo, bits=bits))) <= expected


@pytest.mark.parametrize("gamma", [0, -1])
def test_gamma_invalid_parameters_raise(gamma, small_photo):
    with pytest.raises(ValueError):
        run("Gamma_Correction", small_photo, gamma=gamma)


@pytest.mark.parametrize("bits", [0, 9, -1])
def test_bit_depth_invalid_parameters_raise(bits, small_photo):
    with pytest.raises(ValueError):
        run("Bit_Depth_Reduction", small_photo, bits=bits)


@pytest.mark.parametrize("colors", [2, 4, 8, 16])
@pytest.mark.parametrize("method", ["median_cut", "kmeans"])
def test_quantization_palette_size(colors, method, photo):
    result = run("Color_Quantization", photo, colors=colors, method=method, seed=1)
    assert len(np.unique(result.reshape(-1, 3), axis=0)) <= colors


@pytest.mark.parametrize("levels", [2, 3, 4])
@pytest.mark.parametrize("method", ["floyd_steinberg", "ordered"])
def test_dithering_level_count(levels, method, photo):
    assert len(np.unique(run("Dithering", photo, levels=levels, method=method))) <= levels


@pytest.mark.parametrize("method", ["floyd_steinberg", "ordered"])
def test_dithering_preserves_mean(method, photo):
    """Дизеринг компенсирует потерю градаций шумом, поэтому средняя яркость сохраняется."""
    result = run("Dithering", photo, levels=2, method=method)
    assert abs(result.astype(np.float64).mean() - photo.astype(np.float64).mean()) < 12


@pytest.mark.parametrize("method", ["global", "clahe"])
def test_histogram_equalization_widens_range(method, photo):
    """Эквализация растягивает гистограмму, поэтому разброс яркости не падает."""
    result = run("Histogram_Equalization", photo, method=method)
    assert result.astype(np.float64).std() >= photo.astype(np.float64).std() * 0.9


@pytest.mark.parametrize(
    "name, params",
    [
        ("Color_Quantization", {"colors": 1}),
        ("Color_Quantization", {"method": "kohonen"}),
        ("Color_Quantization", {"method": "kmeans", "colors": 600, "sample_size": 100}),
        ("Dithering", {"levels": 1}),
        ("Dithering", {"method": "ordered", "matrix_size": 1}),
        ("Dithering", {"method": "bayer"}),
        ("Histogram_Equalization", {"method": "adaptive"}),
        ("Histogram_Equalization", {"method": "clahe", "tiles": 1}),
        ("Histogram_Equalization", {"method": "clahe", "bins": 300}),
    ],
)
def test_cython_attacks_reject_invalid_parameters(name, params, photo):
    with pytest.raises(ValueError):
        run(name, photo, **params)
