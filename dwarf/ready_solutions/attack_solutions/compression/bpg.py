"""Атака сжатия BPG: кодек HEVC через внешние утилиты bpgenc и bpgdec."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from dwarf.core.attack_orchestrator.attack_core import Ready_Compression_Attacks
from dwarf.ready_solutions.utils.attack_utils import to_array, to_pil


class Bpg(Ready_Compression_Attacks):
    """
    Атака сжатия BPG (Better Portable Graphics) на базе HEVC.

    В Pillow формат не поддерживается, используются внешние утилиты bpgenc и
    bpgdec из libbpg. Через pip формат недоступен, бинарные файлы должны лежать в PATH.
    """

    @staticmethod
    def attack(**args):
        """
        Пережимает изображение кодеком BPG.

        Кодек работает только с файлами, поэтому обмен идёт через временный
        каталог: он удаляется вместе с промежуточными PNG до возврата результата.

        Args:
            args (dict): параметры атаки
                input_image (np.ndarray): матрица изображения
                quality (int): параметр квантования QP, 0..51, меньше значение — лучше качество (по умолчанию 29)

        Returns:
            np.ndarray: матрица изображения после атаки

        Raises:
            RuntimeError: если bpgenc или bpgdec недоступны в PATH
        """
        defaults = {"input_image": None, "quality": 29}
        args = {**defaults, **args}
        input_image = args["input_image"]
        quality = int(args["quality"])

        if shutil.which("bpgenc") is None or shutil.which("bpgdec") is None:
            raise RuntimeError(
                "Bpg attack requires the bpgenc and bpgdec utilities from libbpg in PATH. "
                "The BPG format is not available via pip."
            )

        with tempfile.TemporaryDirectory() as tmp:
            source_png = str(Path(tmp) / "source.png")
            encoded = str(Path(tmp) / "encoded.bpg")
            decoded_png = str(Path(tmp) / "decoded.png")
            to_pil(input_image).save(source_png)
            subprocess.run(["bpgenc", "-q", str(quality), "-o", encoded, source_png], check=True)
            subprocess.run(["bpgdec", "-o", decoded_png, encoded], check=True)
            return to_array(Image.open(decoded_png))
