import numpy as np

from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
from ...core.embedding_orchestrator.embedding_core import Embedding_Core, Ready_Spatial_Embeddings, Ready_Frequency_Embeddings, Ready_Quantization_Based_Embeddings, Ready_Spread_Spectrum_Embeddings, Ready_SOTA_Embeddings


def load_luma(image_path: str) -> tuple:
    """
    Открывает изображение и разделяет его на яркость и цветность.

    Частотные схемы встраивания работают только с каналом Y: правки в Cb и Cr
    заметнее глазу при той же устойчивости и вдобавок уничтожаются
    субдискретизацией цветности в JPEG. Каналы Cb и Cr возвращаются нетронутыми,
    чтобы save_luma собрал из них исходный кадр.

    Args:
        image_path (str): путь к изображению

    Returns:
        luma (np.ndarray): канал Y, форма (H, W), тип float64, C-совместимый
        chroma (np.ndarray): каналы Cb и Cr, форма (H, W, 2), тип uint8
    """
    channels = np.asarray(Image.open(image_path).convert("YCbCr"), dtype=np.uint8)
    luma = np.ascontiguousarray(channels[:, :, 0], dtype=np.float64)
    chroma = np.ascontiguousarray(channels[:, :, 1:])
    return luma, chroma


def save_luma(luma: np.ndarray, chroma: np.ndarray, output_path: str) -> None:
    """
    Собирает изображение из изменённой яркости и исходной цветности.

    Args:
        luma (np.ndarray): канал Y, форма (H, W), любой вещественный тип
        chroma (np.ndarray): каналы Cb и Cr, форма (H, W, 2), тип uint8
        output_path (str): путь для сохранения результата

    Returns:
        None

    Raises:
        ValueError: если формы яркости и цветности не совпадают
    """
    if luma.shape != chroma.shape[:2]:
        raise ValueError(
            f"Форма яркости {luma.shape} не совпадает с формой цветности {chroma.shape[:2]}"
        )
    ycbcr = np.empty((luma.shape[0], luma.shape[1], 3), dtype=np.uint8)
    ycbcr[:, :, 0] = np.clip(luma, 0, 255).round().astype(np.uint8)
    ycbcr[:, :, 1:] = chroma
    Image.fromarray(ycbcr, "YCbCr").convert("RGB").save(output_path)


def bits_to_array(watermark_bits) -> np.ndarray:
    """
    Приводит ЦВЗ к виду, который принимают Cython-расширения.

    Строка вида '1011' - основной формат ЦВЗ в фреймворке: его возвращает
    extraction и его же ждёт метрика BER. Расширениям нужен C-совместимый
    массив int32, поэтому преобразование вынесено сюда, а не размножено
    по классам встраивания.

    Args:
        watermark_bits (str | Sequence[int] | np.ndarray): биты ЦВЗ

    Returns:
        np.ndarray: биты ЦВЗ, форма (L,), тип int32, C-совместимый

    Raises:
        ValueError: если ЦВЗ пуст, многомерен или содержит значения кроме 0 и 1
    """
    if isinstance(watermark_bits, str):
        source = [int(symbol) for symbol in watermark_bits.strip() if not symbol.isspace()]
    else:
        source = watermark_bits

    array = np.ascontiguousarray(source, dtype=np.int32)

    if array.ndim != 1:
        raise ValueError(f"ЦВЗ должен быть одномерным, получена форма {array.shape}")
    if array.size == 0:
        raise ValueError("Пустой ЦВЗ")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("ЦВЗ должен состоять только из 0 и 1")

    return array


def array_to_bits(watermark: np.ndarray) -> str:
    """
    Переводит извлечённый ЦВЗ обратно в строку из '0' и '1'.

    Args:
        watermark (np.ndarray): биты ЦВЗ, форма (L,), любой целочисленный тип

    Returns:
        str: строка длины L из символов '0' и '1'
    """
    return "".join("1" if bit else "0" for bit in watermark)
