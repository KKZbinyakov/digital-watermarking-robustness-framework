from ...utils.attack_utils import *


class Bpg(Ready_Compression_Attacks):
    """
    Атака сжатия BPG (Better Portable Graphics) на базе HEVC.

    В Pillow формат не поддерживается, используются внешние утилиты bpgenc и
    bpgdec из libbpg. Через pip формат недоступен, биные файлы должны лежать в PATH.
    """

    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Пережимает изображение кодеком BPG и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                quality (int): параметр квантования QP, 0..51, меньше значение — лучше качество (по умолчанию 29)

        Returns:
            None

        Raises:
            RuntimeError: если bpgenc или bpgdec недоступны в PATH
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        quality = int(args.get("quality", 29))

        if shutil.which("bpgenc") is None or shutil.which("bpgdec") is None:
            raise RuntimeError(
                "Для атаки Bpg нужны утилиты bpgenc и bpgdec из libbpg в PATH."
                "Через pip формат BPG недоступен."
            )

        img = Image.open(input_data).convert("RGB")
        with tempfile.TemporaryDirectory() as tmp:
            source_png = str(Path(tmp) / "source.png")
            encoded = str(Path(tmp) / "encoded.bpg")
            decoded_png = str(Path(tmp) / "decoded.png")
            img.save(source_png)
            subprocess.run(["bpgenc", "-q", str(quality), "-o", encoded, source_png], check=True)
            subprocess.run(["bpgdec", "-o", decoded_png, encoded], check=True)
            Image.open(decoded_png).convert("RGB").save(output_data)
