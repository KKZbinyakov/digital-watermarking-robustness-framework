"""Тесты метрик экспертизы.

Помимо общего контракта здесь проверяются точные значения там, где их можно
вывести аналитически, и качественные свойства там, где нельзя: монотонность по
силе искажения, пределы на совпадающих входах, согласованность метрик между
собой. Такие проверки ловят ошибки, которые контрактные тесты пропускают.
"""

import numpy as np
import pytest
from conftest import make_photo, solutions

import dwarf.ready_solutions.expertise_solutions  # noqa: F401  наполняет реестр метрик
from dwarf.core.expertise_orchestrator.expertise_core import Expertise_Core

METRICS = solutions(Expertise_Core.get_registered_expertises())
NAMES = sorted(METRICS)

FULL_REFERENCE = frozenset({"MSE", "PSNR", "SSIM", "MS_SSIM", "FSIM", "FSIMc", "VIF", "LPIPS", "DISTS"})
"""Метрики, сравнивающие оригинал с искажённым изображением."""

NO_REFERENCE = frozenset({"NIQE", "BRISQUE"})
"""Безэталонные метрики: оценивают одно изображение."""

BIT_STRINGS = frozenset({"BER", "NC"})
"""Метрики, сравнивающие битовые строки ЦВЗ."""

NEURAL = frozenset({"LPIPS", "DISTS", "NIQE", "BRISQUE"})
"""Метрики на pyiqa: пропускаются, если пакет не установлен."""

IDENTICAL_LIMIT = {
    "MSE": 0.0,
    "PSNR": float("inf"),
    "SSIM": 1.0,
    "MS_SSIM": 1.0,
    "FSIM": 1.0,
    "FSIMc": 1.0,
    "VIF": 1.0,
}
"""Значение метрики на паре одинаковых изображений."""

MINIMUM_SIDE = {"MS_SSIM": 176, "VIF": 72}
"""Метрики, требующие кадра не меньше указанной стороны."""


def evaluate(name, **arguments):
    """
    Вызывает метрику, пропуская тест при отсутствии необязательной зависимости.

    Args:
        name (str): имя метрики в реестре
        arguments: аргументы метрики

    Returns:
        float: значение метрики
    """
    try:
        return Expertise_Core.get_expertise_class_by_name(name).expertise(**arguments)
    except RuntimeError as error:
        pytest.skip(f"{name}: {error}")


def arguments_for(name, original, distorted):
    """
    Собирает аргументы метрики по её семейству.

    Args:
        name (str): имя метрики
        original (np.ndarray): матрица оригинала
        distorted (np.ndarray): матрица искажённого изображения

    Returns:
        dict: аргументы для вызова метрики
    """
    if name in NO_REFERENCE:
        return {"input_image": distorted}
    if name in FULL_REFERENCE:
        return {"original_image": original, "distorted_image": distorted}
    if name in BIT_STRINGS:
        return {"original_bits": "1011001010", "extracted_bits": "1011001011"}
    if name == "Watermark_PSNR":
        return {"original_watermark": original, "extracted_watermark": distorted}
    if name == "AUC":
        return {"y_true": np.array([1, 0, 1, 0]), "y_scores": np.array([0.9, 0.1, 0.8, 0.2])}
    if name == "P_Value":
        return {"statistic": 2.0, "null_samples": np.array([0.0, 1.0, 3.0])}
    return {"y_true": np.array([1, 0, 1, 0]), "y_pred": np.array([1, 0, 0, 0])}


@pytest.fixture
def distorted(photo):
    """Оригинал, испорченный гауссовым шумом."""
    noise = np.random.default_rng(99).normal(0.0, 10.0, photo.shape)
    return np.clip(photo.astype(np.float64) + noise, 0, 255).astype(np.uint8)


# --- общий контракт -------------------------------------------------------


def test_registry_is_not_empty():
    assert METRICS, "реестр метрик пуст: категории не импортировались"


@pytest.mark.parametrize("name", NAMES)
def test_reachable_through_orchestrator(name):
    assert Expertise_Core.get_expertise_class_by_name(name) is getattr(Expertise_Core, name)


@pytest.mark.parametrize("name", NAMES)
def test_returns_float(name, photo, distorted):
    value = evaluate(name, **arguments_for(name, photo, distorted))
    assert isinstance(value, float), f"{name} вернула {type(value).__name__}, а не число"
    assert not np.isnan(value), f"{name} вернула nan на корректных данных"


@pytest.mark.parametrize("name", NAMES)
def test_deterministic(name, photo, distorted):
    arguments = arguments_for(name, photo, distorted)
    assert evaluate(name, **arguments) == evaluate(name, **arguments)


@pytest.mark.parametrize("name", NAMES)
def test_does_not_modify_inputs(name, photo, distorted):
    before_original, before_distorted = photo.copy(), distorted.copy()
    evaluate(name, **arguments_for(name, photo, distorted))
    assert np.array_equal(photo, before_original)
    assert np.array_equal(distorted, before_distorted)


@pytest.mark.parametrize("name", sorted(IDENTICAL_LIMIT))
def test_identical_images_reach_limit(name, photo):
    value = evaluate(name, original_image=photo, distorted_image=photo)
    expected = IDENTICAL_LIMIT[name]
    if expected in (0.0, float("inf")):
        assert value == expected
    else:
        assert value == pytest.approx(expected, abs=1e-6)


REQUIRE_RGB = frozenset({"MSE", "PSNR", "FSIMc"})
"""Метрики, которым нужны все три канала: одноканальную матрицу они обязаны отвергать."""

ACCEPT_GRAY = frozenset({"SSIM", "MS_SSIM", "FSIM", "VIF"})
"""Метрики по яркости: одноканальная матрица для них — уже готовая яркость."""


@pytest.mark.parametrize("name", sorted(REQUIRE_RGB))
def test_rejects_single_channel_matrix(name, photo):
    with pytest.raises(ValueError):
        evaluate(name, original_image=np.zeros(photo.shape[:2]), distorted_image=photo)


@pytest.mark.parametrize("name", sorted(ACCEPT_GRAY))
def test_accepts_single_channel_matrix(name, photo):
    """Яркостные метрики принимают одноканальный кадр и дают предел на совпадающей паре."""
    luma = 0.299 * photo[..., 0] + 0.587 * photo[..., 1] + 0.114 * photo[..., 2]
    assert evaluate(name, original_image=luma, distorted_image=luma) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", sorted(ACCEPT_GRAY))
def test_gray_and_rgb_inputs_agree(name, photo, distorted):
    """Матрица RGB и её же яркость обязаны давать одно значение: метрика всё равно считает по яркости."""
    luma = lambda a: 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]  # noqa: E731
    from_rgb = evaluate(name, original_image=photo, distorted_image=distorted)
    from_gray = evaluate(name, original_image=luma(photo), distorted_image=luma(distorted))
    assert from_rgb == pytest.approx(from_gray)


@pytest.mark.parametrize("name", sorted((FULL_REFERENCE | {"Watermark_PSNR"}) - NEURAL))
def test_rejects_malformed_matrix(name, photo):
    with pytest.raises(ValueError):
        evaluate(name, **arguments_for(name, np.zeros((4, 4, 4, 4)), photo))


# --- точные значения ------------------------------------------------------


@pytest.mark.parametrize("offset", [1, 4, 16, 64])
def test_mse_matches_closed_form(offset, photo):
    """Сдвиг всех уровней на константу даёт MSE, равный квадрату сдвига."""
    shifted = np.clip(photo.astype(np.int16) - offset, 0, 255).astype(np.uint8)
    difference = photo.astype(np.float64) - shifted.astype(np.float64)
    expected = float(np.mean(difference**2))
    assert evaluate("MSE", original_image=photo, distorted_image=shifted) == pytest.approx(expected)


@pytest.mark.parametrize("offset", [1, 4, 16, 64])
def test_psnr_matches_closed_form(offset, photo):
    shifted = np.clip(photo.astype(np.int16) - offset, 0, 255).astype(np.uint8)
    mse = evaluate("MSE", original_image=photo, distorted_image=shifted)
    expected = 10 * np.log10(255**2 / mse)
    assert evaluate("PSNR", original_image=photo, distorted_image=shifted) == pytest.approx(expected)


def test_psnr_and_mse_agree(photo, distorted):
    mse = evaluate("MSE", original_image=photo, distorted_image=distorted)
    psnr = evaluate("PSNR", original_image=photo, distorted_image=distorted)
    assert psnr == pytest.approx(10 * np.log10(255**2 / mse))


def test_ssim_matches_skimage(photo, distorted):
    skimage_metrics = pytest.importorskip("skimage.metrics")
    luma = lambda a: 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]  # noqa: E731
    expected = skimage_metrics.structural_similarity(
        luma(photo.astype(np.float64)),
        luma(distorted.astype(np.float64)),
        data_range=255,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    assert evaluate("SSIM", original_image=photo, distorted_image=distorted) == pytest.approx(expected)


# --- монотонность ---------------------------------------------------------


@pytest.mark.parametrize("name", sorted((FULL_REFERENCE | {"Watermark_PSNR"}) - NEURAL - {"MSE"}))
def test_decreases_with_distortion(name, photo):
    """Чем сильнее шум, тем ниже значение метрики сходства."""
    values = []
    for sigma in (2.0, 8.0, 20.0, 40.0):
        noise = np.random.default_rng(5).normal(0.0, sigma, photo.shape)
        noisy = np.clip(photo.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        values.append(evaluate(name, **arguments_for(name, photo, noisy)))
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1)), values


def test_mse_grows_with_distortion(photo):
    values = []
    for sigma in (2.0, 8.0, 20.0, 40.0):
        noise = np.random.default_rng(5).normal(0.0, sigma, photo.shape)
        noisy = np.clip(photo.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        values.append(evaluate("MSE", original_image=photo, distorted_image=noisy))
    assert all(values[i] < values[i + 1] for i in range(len(values) - 1)), values


@pytest.mark.parametrize("name", sorted(MINIMUM_SIDE))
def test_small_frames_raise(name):
    side = MINIMUM_SIDE[name] - 1
    tiny = make_photo(side, side, seed=2)
    with pytest.raises(ValueError):
        evaluate(name, original_image=tiny, distorted_image=tiny)


# --- BER и NC -------------------------------------------------------------


@pytest.mark.parametrize(
    "original_bits, extracted_bits, expected",
    [
        ("1010", "1010", 0.0),
        ("1010", "1011", 0.25),
        ("1010", "1001", 0.5),
        ("1010", "0101", 1.0),
        ("11111111", "00000000", 1.0),
        ("11111111", "11111110", 0.125),
        ("1100110011001100", "1100110011001101", 0.0625),
        ("1" * 64, "0" * 64, 1.0),
        ("1" * 64, "1" * 63 + "0", 1.0 / 64),
        ("1" * 64, "0" * 32 + "1" * 32, 0.5),
    ],
)
def test_ber_exact_values(original_bits, extracted_bits, expected):
    """Регрессия: сравнение строк вместо массивов давало 1/длина при любом числе ошибок."""
    assert evaluate("BER", original_bits=original_bits, extracted_bits=extracted_bits) == pytest.approx(expected)


@pytest.mark.parametrize("length", [8, 16, 64, 256])
def test_ber_reaches_one_on_full_inversion(length):
    """Полная инверсия обязана давать ровно единицу независимо от длины ЦВЗ."""
    original_bits = "10" * (length // 2)
    inverted = "".join("1" if bit == "0" else "0" for bit in original_bits)
    assert evaluate("BER", original_bits=original_bits, extracted_bits=inverted) == pytest.approx(1.0)


@pytest.mark.parametrize("errors", [0, 1, 7, 16, 31, 32])
def test_ber_counts_every_error(errors):
    """BER обязан расти с числом ошибок, а не только реагировать на сам факт различия."""
    length = 32
    original_bits = "0" * length
    extracted_bits = "1" * errors + "0" * (length - errors)
    assert evaluate("BER", original_bits=original_bits, extracted_bits=extracted_bits) == pytest.approx(errors / length)


@pytest.mark.parametrize(
    "original_bits, extracted_bits",
    [("1010", "1010"), ("1010", "1011"), ("1010", "1001"), ("1010", "0101"), ("11001010", "10101100")],
)
def test_nc_agrees_with_ber(original_bits, extracted_bits):
    """Для строк одинаковой длины NC = 1 - 2 * BER: метрики обязаны быть согласованы."""
    ber = evaluate("BER", original_bits=original_bits, extracted_bits=extracted_bits)
    nc = evaluate("NC", original_bits=original_bits, extracted_bits=extracted_bits)
    assert nc == pytest.approx(1 - 2 * ber)


def test_nc_changes_sign_on_inversion():
    original_bits = "11010010"
    inverted = "".join("1" if bit == "0" else "0" for bit in original_bits)
    assert evaluate("NC", original_bits=original_bits, extracted_bits=original_bits) == pytest.approx(1.0)
    assert evaluate("NC", original_bits=original_bits, extracted_bits=inverted) == pytest.approx(-1.0)


@pytest.mark.parametrize("name", sorted(BIT_STRINGS))
def test_length_mismatch_raises_by_default(name):
    with pytest.raises(ValueError):
        evaluate(name, original_bits="1010", extracted_bits="10101")


@pytest.mark.parametrize("name", sorted(BIT_STRINGS))
def test_length_mismatch_allowed_explicitly(name):
    value = evaluate(name, original_bits="1010", extracted_bits="10101", allow_length_mismatch=True)
    assert isinstance(value, float)


@pytest.mark.parametrize("name", sorted(BIT_STRINGS))
def test_empty_bit_strings_raise(name):
    with pytest.raises(ValueError):
        evaluate(name, original_bits="", extracted_bits="")


@pytest.mark.parametrize("name", sorted(BIT_STRINGS))
@pytest.mark.parametrize("bits", ["10x1", "1 01", "abcd"])
def test_non_binary_characters_raise(name, bits):
    """Посторонний символ раньше молча считался нулевым битом."""
    with pytest.raises(ValueError):
        evaluate(name, original_bits=bits, extracted_bits="1011")


# --- детекторные метрики --------------------------------------------------


def test_detector_metrics_match_closed_form(detector):
    y_true, y_pred = detector["y_true"], detector["y_pred"]
    true_positive = int(((y_pred == 1) & (y_true == 1)).sum())
    true_negative = int(((y_pred == 0) & (y_true == 0)).sum())
    false_positive = int(((y_pred == 1) & (y_true == 0)).sum())
    false_negative = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    expected = {
        "Accuracy": (true_positive + true_negative) / y_true.size,
        "Precision": precision,
        "Recall": recall,
        "F1": 2 * precision * recall / (precision + recall),
    }
    for name, value in expected.items():
        assert evaluate(name, y_true=y_true, y_pred=y_pred) == pytest.approx(value), name


def test_auc_matches_pairwise_definition(detector):
    y_true, y_scores = detector["y_true"], detector["y_scores"]
    positives, negatives = y_scores[y_true == 1], y_scores[y_true == 0]
    expected = float(np.mean([(p > n) + 0.5 * (p == n) for p in positives for n in negatives]))
    assert evaluate("AUC", y_true=y_true, y_scores=y_scores) == pytest.approx(expected)


def test_auc_of_perfect_detector_is_one():
    assert evaluate("AUC", y_true=np.array([0, 0, 1, 1]), y_scores=np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_auc_of_constant_scores_is_half():
    """При полностью совпадающих оценках усреднение рангов обязано дать ровно 0.5."""
    assert evaluate("AUC", y_true=np.array([1, 0, 1, 0]), y_scores=np.array([0.5] * 4)) == pytest.approx(0.5)


def test_auc_is_nan_when_one_class_missing():
    assert np.isnan(evaluate("AUC", y_true=np.array([1, 1, 1]), y_scores=np.array([0.1, 0.5, 0.9])))


@pytest.mark.parametrize("size", [1, 10, 1000])
def test_p_value_never_reaches_zero(size):
    """Поправка на единицу не даёт утверждать невозможность события по конечной выборке."""
    value = evaluate("P_Value", statistic=1e9, null_samples=np.zeros(size))
    assert value == pytest.approx(1 / (size + 1)) and value > 0


def test_p_value_of_typical_statistic_is_near_one():
    assert evaluate("P_Value", statistic=-1e9, null_samples=np.zeros(10)) == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["Accuracy", "Precision", "Recall", "F1"])
def test_label_shape_mismatch_raises(name):
    with pytest.raises(ValueError):
        evaluate(name, y_true=np.array([1, 0, 1]), y_pred=np.array([1, 0]))


@pytest.mark.parametrize("name", ["Accuracy", "Precision", "Recall", "F1"])
def test_empty_labels_raise(name):
    with pytest.raises(ValueError):
        evaluate(name, y_true=np.array([]), y_pred=np.array([]))


# --- изображение-знак -----------------------------------------------------


def test_watermark_psnr_is_infinite_on_identical(photo):
    assert evaluate("Watermark_PSNR", original_watermark=photo, extracted_watermark=photo) == float("inf")


def test_watermark_psnr_accepts_single_channel():
    """Знаки часто хранятся одноканальными, метрика обязана их принимать."""
    watermark = (np.random.default_rng(3).random((32, 32)) > 0.5).astype(np.uint8) * 255
    assert evaluate("Watermark_PSNR", original_watermark=watermark, extracted_watermark=watermark) == float("inf")


def test_watermark_psnr_matches_between_channel_layouts():
    watermark = (np.random.default_rng(3).random((32, 32)) > 0.5).astype(np.uint8) * 255
    noisy = np.clip(watermark.astype(np.float64) + 20, 0, 255).astype(np.uint8)
    single = evaluate("Watermark_PSNR", original_watermark=watermark, extracted_watermark=noisy)
    triple = evaluate(
        "Watermark_PSNR",
        original_watermark=np.stack([watermark] * 3, axis=-1),
        extracted_watermark=np.stack([noisy] * 3, axis=-1),
    )
    assert single == pytest.approx(triple)
