"""Тесты атак.

Проверяются две вещи: общий контракт, одинаковый для всех атак, и частные
свойства отдельных атак. Общая часть параметризуется прямо из реестра, поэтому
новая атака попадает под проверки, ничего не дописывая сюда.
"""

import numpy as np
import pytest
from conftest import make_photo, solutions
from PIL import Image

import dwarf.ready_solutions.attack_solutions  # noqa: F401  наполняет реестр атак
from dwarf.core.attack_orchestrator.attack_core import Attack_Core

ATTACKS = solutions(Attack_Core.get_registered_attacks())
NAMES = sorted(ATTACKS)

LOSSLESS = frozenset({"Tiff", "Flif"})
"""Атаки без потерь: обязаны возвращать матрицу, совпадающую с входом побитово."""

SEEDED = frozenset({
    "Color_Jitter",
    "Color_Space_Noise",
    "AWGN",
    "Impulse",
    "Periodic",
    "Poisson",
    "Salt_and_Pepper",
    "Speckle",
})
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


def mean_absolute_error(image, reference):
    """
    Средняя абсолютная разница по пикселям — грубая мера силы искажения.

    Args:
        image (np.ndarray): матрица изображения после атаки
        reference (np.ndarray): исходная матрица изображения

    Returns:
        float: средняя абсолютная разница уровней
    """
    return np.abs(image.astype(np.float64) - reference.astype(np.float64)).mean()


def changed_pixel_count(image, reference):
    """
    Число пикселей, изменившихся хотя бы по одному каналу.

    Args:
        image (np.ndarray): матрица изображения после атаки
        reference (np.ndarray): исходная матрица изображения

    Returns:
        int: число различающихся пикселей
    """
    return int(np.count_nonzero(np.any(image != reference, axis=2)))


def high_frequency_energy(image):
    """
    Грубая мера мелкой структуры: средний модуль разности соседних пикселей.

    Размытие и диффузия эту величину снижают, увеличение резкости — поднимают.

    Args:
        image (np.ndarray): матрица изображения

    Returns:
        float: сумма средних модулей разности по строкам и по столбцам
    """
    data = image.astype(np.float64)
    return np.abs(np.diff(data, axis=1)).mean() + np.abs(np.diff(data, axis=0)).mean()


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


# --- шум --------------------------------------------------------------


@pytest.mark.parametrize("weak, strong", [(0.01, 0.08)])
def test_awgn_higher_sigma_distorts_more(weak, strong, photo):
    error_weak = mean_absolute_error(run("AWGN", photo, sigma=weak, seed=1), photo)
    error_strong = mean_absolute_error(run("AWGN", photo, sigma=strong, seed=1), photo)
    assert error_strong > error_weak


@pytest.mark.parametrize("low, high", [(0.001, 0.05)])
def test_impulse_higher_density_corrupts_more_pixels(low, high, photo):
    changed_low = changed_pixel_count(run("Impulse", photo, density=low, seed=1), photo)
    changed_high = changed_pixel_count(run("Impulse", photo, density=high, seed=1), photo)
    assert changed_high > changed_low


@pytest.mark.parametrize("weak, strong", [(0.01, 0.2)])
def test_periodic_higher_amplitude_distorts_more(weak, strong, photo):
    error_weak = mean_absolute_error(run("Periodic", photo, amplitude=weak, seed=1), photo)
    error_strong = mean_absolute_error(run("Periodic", photo, amplitude=strong, seed=1), photo)
    assert error_strong > error_weak


@pytest.mark.parametrize("low_peak, high_peak", [(2.0, 500.0)])
def test_poisson_lower_peak_distorts_more(low_peak, high_peak, photo):
    """Меньше peak значит меньше фотонов на пиксель и относительно сильнее дробовой шум."""
    error_low_peak = mean_absolute_error(run("Poisson", photo, peak=low_peak, seed=1), photo)
    error_high_peak = mean_absolute_error(run("Poisson", photo, peak=high_peak, seed=1), photo)
    assert error_low_peak > error_high_peak


@pytest.mark.parametrize("low, high", [(0.001, 0.05)])
def test_salt_and_pepper_higher_density_corrupts_more_pixels(low, high, photo):
    changed_low = changed_pixel_count(run("Salt_and_Pepper", photo, density=low, seed=1), photo)
    changed_high = changed_pixel_count(run("Salt_and_Pepper", photo, density=high, seed=1), photo)
    assert changed_high > changed_low


def test_salt_and_pepper_only_produces_pure_black_or_white(photo):
    result = run("Salt_and_Pepper", photo, density=0.05, seed=1)
    changed_mask = np.any(result != photo, axis=2)
    changed_pixels = result[changed_mask]
    is_salt = np.all(changed_pixels == 255, axis=1)
    is_pepper = np.all(changed_pixels == 0, axis=1)
    assert np.all(is_salt | is_pepper)


@pytest.mark.parametrize("weak, strong", [(0.001, 0.05)])
def test_speckle_higher_variance_distorts_more(weak, strong, photo):
    error_weak = mean_absolute_error(run("Speckle", photo, variance=weak, seed=1), photo)
    error_strong = mean_absolute_error(run("Speckle", photo, variance=strong, seed=1), photo)
    assert error_strong > error_weak


# --- фильтрация ---------------------------------------------------------


@pytest.mark.parametrize("weak, strong", [(0.5, 5.0)])
def test_gaussian_blur_higher_sigma_smooths_more(weak, strong, photo):
    energy_weak = high_frequency_energy(run("Gaussian_Blur", photo, sigma=weak))
    energy_strong = high_frequency_energy(run("Gaussian_Blur", photo, sigma=strong))
    assert energy_strong < energy_weak


@pytest.mark.parametrize("small, large", [(3, 15)])
def test_box_filter_larger_window_smooths_more(small, large, photo):
    energy_small = high_frequency_energy(run("Box_Filter", photo, window=small))
    energy_large = high_frequency_energy(run("Box_Filter", photo, window=large))
    assert energy_large < energy_small


@pytest.mark.parametrize("few, many", [(1, 50)])
def test_anisotropic_diffusion_more_iterations_smooths_more(few, many, photo):
    energy_few = high_frequency_energy(run("Anisotropic_Diffusion", photo, iterations=few))
    energy_many = high_frequency_energy(run("Anisotropic_Diffusion", photo, iterations=many))
    assert energy_many < energy_few


def test_unsharp_mask_sharpens_by_default(photo):
    assert high_frequency_energy(run("Unsharp_Mask", photo)) > high_frequency_energy(photo)


@pytest.mark.parametrize("weak, strong", [(0.2, 4.0)])
def test_unsharp_mask_stronger_amount_sharpens_more(weak, strong, photo):
    energy_weak = high_frequency_energy(run("Unsharp_Mask", photo, amount=weak))
    energy_strong = high_frequency_energy(run("Unsharp_Mask", photo, amount=strong))
    assert energy_strong > energy_weak


def test_median_filter_reduces_salt_and_pepper_noise(photo):
    noisy = run("Salt_and_Pepper", photo, density=0.05, seed=1)
    cleaned = run("Median_Filter", noisy, window=3)
    assert mean_absolute_error(cleaned, photo) < mean_absolute_error(noisy, photo)


def test_wiener_filter_reduces_gaussian_noise(photo):
    noisy = run("AWGN", photo, sigma=0.05, seed=1)
    cleaned = run("Wiener_Filter", noisy)
    assert mean_absolute_error(cleaned, photo) < mean_absolute_error(noisy, photo)


def test_bilateral_filter_reduces_gaussian_noise(photo):
    noisy = run("AWGN", photo, sigma=0.05, seed=1)
    cleaned = run("Bilateral_Filter", noisy)
    assert mean_absolute_error(cleaned, photo) < mean_absolute_error(noisy, photo)


def test_homomorphic_filter_preserves_chroma(photo):
    """Фильтруется только канал Y, поэтому цветность меняется лишь на погрешность round-trip YCbCr."""
    result = run("Homomorphic_Filter", photo)
    original_ycbcr = np.asarray(Image.fromarray(photo, "RGB").convert("YCbCr"), dtype=np.float64)
    result_ycbcr = np.asarray(Image.fromarray(result, "RGB").convert("YCbCr"), dtype=np.float64)
    assert mean_absolute_error(result_ycbcr[..., 1:], original_ycbcr[..., 1:]) < 2.0
    assert not np.array_equal(result_ycbcr[..., 0], original_ycbcr[..., 0])


@pytest.mark.parametrize("mild, strong", [(1.2, 3.0)])
def test_homomorphic_filter_higher_gamma_high_boosts_detail(mild, strong, photo):
    energy_mild = high_frequency_energy(run("Homomorphic_Filter", photo, gamma_high=mild))
    energy_strong = high_frequency_energy(run("Homomorphic_Filter", photo, gamma_high=strong))
    assert energy_strong > energy_mild


@pytest.mark.parametrize("level", [0, 64, 128, 200, 255])
def test_homomorphic_filter_keeps_uniform_frame(level):
    """
    Однотонный кадр обязан остаться однотонным и того же уровня.

    Усиление в логарифмической области возвращается к масштабу входа по средней
    яркости. Растягивание по минимуму и максимуму обращало бы такой кадр в чёрный:
    у него минимум равен максимуму.
    """
    uniform = np.full((32, 32, 3), level, dtype=np.uint8)
    result = run("Homomorphic_Filter", uniform)
    assert mean_absolute_error(result, uniform) < 1.5


@pytest.mark.parametrize("spike", [130, 180, 255])
def test_homomorphic_filter_ignores_single_outlier(spike, photo):
    """Один выброс не должен менять яркость всего кадра: нормировка идёт по среднему, а не по краям гистограммы."""
    spiked = photo.copy()
    spiked[0, 0] = spike

    baseline = run("Homomorphic_Filter", photo).astype(np.float64)
    result = run("Homomorphic_Filter", spiked).astype(np.float64)

    assert abs(result.mean() - baseline.mean()) < 0.5


@pytest.mark.parametrize("c", [0.0, -5.0, 10.001])
def test_homomorphic_filter_rejects_invalid_steepness(c, small_photo):
    with pytest.raises(ValueError):
        run("Homomorphic_Filter", small_photo, c=c)


@pytest.mark.parametrize("weak, strong", [(100.0, 10.0)])
def test_homomorphic_filter_lower_cutoff_distorts_more(weak, strong, photo):
    """Чем ниже частота среза, тем большая часть спектра попадает под усиление gamma_high."""
    error_weak = mean_absolute_error(run("Homomorphic_Filter", photo, cutoff=weak), photo)
    error_strong = mean_absolute_error(run("Homomorphic_Filter", photo, cutoff=strong), photo)
    assert error_strong > error_weak
