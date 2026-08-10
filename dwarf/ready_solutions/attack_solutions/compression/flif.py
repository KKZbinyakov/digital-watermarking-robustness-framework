"""Атака перекодирования в FLIF: сжатие без потерь через внешнюю утилиту flif."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from dwarf.core.attack_orchestrator.attack_core import Ready_Compression_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_array, to_pil


class Flif(Ready_Compression_Attacks):
    """
    Атака перекодирования в FLIF (Free Lossless Image Format).

    Сжатие без потерь, пиксели не меняются: как и Tiff, атака контрольная и
    обязана давать нулевой BER
    В Pillow не поддерживается, используется внешняя утилита flif.
    """

    @staticmethod
    def attack(**args):
        """
        Перекодирует изображение через FLIF.

        Кодек работает только с файлами, поэтому обмен идёт через временный
        каталог: он удаляется вместе с промежуточными PNG до возврата результата.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            RuntimeError: если утилита flif недоступна в PATH
        """
        defaults = {"input_image": None}
        args = {**defaults, **args}
        input_image = args["input_image"]

        if shutil.which("flif") is None:
            raise RuntimeError(
                "Flif attack requires the flif utility in PATH. "
                "The format is abandoned, consider JPEG XL as a modern replacement."
            )

        with tempfile.TemporaryDirectory() as tmp:
            source_png = str(Path(tmp) / "source.png")
            encoded = str(Path(tmp) / "encoded.flif")
            decoded_png = str(Path(tmp) / "decoded.png")
            to_pil(input_image).save(source_png)
            subprocess.run(["flif", "-e", source_png, encoded], check=True)
            subprocess.run(["flif", "-d", encoded, decoded_png], check=True)
            return to_array(Image.open(decoded_png))
