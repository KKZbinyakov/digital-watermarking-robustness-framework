from pathlib import Path
from typing import Iterator

from ..utils.utils import import_function


# Форматы, которые ядро считает валидными носителями ЦВЗ.
SUPPORTED_EXTENSIONS: tuple = (".png", ".bmp", ".tiff", ".tif", ".ppm")


class Ds_Core:
    """
    Оркестратор датасетов (dataset orchestrator).

    Две задачи:
    1. Регистрация сменных ds-решений
    2. Поставка изображений-носителей в пайплайн
    """

    def __init__(self):
        self.available_ds: dict = {}   # реестр подключённых ds-решений
        self.dataset: list = []        # текущий список путей к изображениям
        pass

    def get_new_ds(self, ds_solutions: dict) -> None:
        """
        Регистрирует новые ds-решения, загружая их по пути к файлу.

        Args:
            ds_solutions (dict): отображение {имя: путь_к_py_файлу}, как в остальных Core.

        Returns:
            None
        """
        for ds in ds_solutions:
            self.available_ds[ds] = import_function(ds, ds_solutions[ds])
        return

    def get_all_available_ds(self) -> None:
        """
        Печатает имена всех зарегистрированных ds-решений.

        Returns:
            None
        """
        for ds in self.available_ds:
            print(ds)
        return

    def load_from_directory(self, directory: str, recursive: bool = False,
                            extensions: tuple = SUPPORTED_EXTENSIONS) -> list:
        """
        Собирает список путей к изображениям-носителям из директории.

        Args:
            directory (str): путь к папке с изображениями.
            recursive (bool): искать ли во вложенных папках.
            extensions (tuple): допустимые расширения (по умолчанию без JPEG).

        Returns:
            list: отсортированный список абсолютных путей к изображениям.
        """
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(f"Не директория: {directory}")
        pattern = "**/*" if recursive else "*"
        exts = {e.lower() for e in extensions}
        self.dataset = sorted(
            str(p.resolve())
            for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in exts
        )
        return self.dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[str]:
        """Итерация по путям текущего датасета"""
        return iter(self.dataset)