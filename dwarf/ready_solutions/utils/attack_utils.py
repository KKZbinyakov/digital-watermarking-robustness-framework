import io
import shutil
import subprocess
import tempfile
 
import numpy as np
import cv2

from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from scipy.signal import wiener
from ...core.attack_orchestrator.attack_core import Attack_Core, Ready_Geometric_Attacks, Ready_Noise_Attacks, Ready_Filtering_Attacks, Ready_Compression_Attacks, Ready_Color_Brightness_Attacks, Ready_SOTA_Attacks, Ready_Watermark_Removal_Networks_Attacks, Ready_Adversarial_Examples_Attacks, Ready_Collage_Synchronization_Attacks, Ready_Physical_Attacks


def load_rgb(image_path: str) -> np.ndarray:
    """
    Открывает изображение и приводит его к трёхканальному RGB.

    Args:
        image_path (str): путь к изображению

    Returns:
        np.ndarray: массив формы (H, W, 3) типа uint8
    """
    return np.asarray(Image.open(image_path).convert("RGB"))


def save_rgb(array: np.ndarray, output_path: str) -> None:
    """
    Сохраняет массив как RGB-изображение с округлением до целых уровней.

    Args:
        array (np.ndarray): массив формы (H, W, 3) любого числового типа
        output_path (str): путь для сохранения результата

    Returns:
        None
    """
    Image.fromarray(np.clip(array, 0, 255).round().astype(np.uint8), "RGB").save(output_path)


def roundtrip_buffer(img: Image.Image, fmt: str, **save_kwargs) -> Image.Image:
    """
    Прогоняет изображение через кодек в памяти: кодирует и сразу декодирует.

    Args:
        img (Image.Image): исходное изображение
        fmt (str): имя формата Pillow, например 'JPEG' или 'WEBP'
        save_kwargs: параметры кодека, передаваемые в Image.save

    Returns:
        Image.Image: раскодированное изображение в режиме 'RGB'
    """
    buffer = io.BytesIO()
    img.save(buffer, format=fmt, **save_kwargs)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")
