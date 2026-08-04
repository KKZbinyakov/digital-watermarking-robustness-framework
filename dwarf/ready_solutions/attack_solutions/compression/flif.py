import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from dwarf.core.attack_orchestrator.attack_core import Ready_Compression_Attacks


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
        Перекодирует изображение через FLIF и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата

        Returns:
            None

        Raises:
            RuntimeError: если утилита flif недоступна в PATH
        """
        defaults = {"input_data": None}
        args = {**defaults, **args}
        input_data = args["input_data"]
        output_data = args["output_data"]

        if shutil.which("flif") is None:
            raise RuntimeError(
                "Для атаки Flif нужна утилита flif в PATH. Формат заброшен, "
                "как современную замену стоит рассмотреть JPEG XL."
            )

        img = Image.open(input_data).convert("RGB")
        with tempfile.TemporaryDirectory() as tmp:
            source_png = str(Path(tmp) / "source.png")
            encoded = str(Path(tmp) / "encoded.flif")
            decoded_png = str(Path(tmp) / "decoded.png")
            img.save(source_png)
            subprocess.run(["flif", "-e", source_png, encoded], check=True)
            subprocess.run(["flif", "-d", encoded, decoded_png], check=True)
            Image.open(decoded_png).convert("RGB").save(output_data)
