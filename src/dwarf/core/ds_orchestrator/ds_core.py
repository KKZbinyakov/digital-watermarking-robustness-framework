from pathlib import Path
from typing import Iterator

from src.dwarf.core.utils.utils import *

class Ds_Core_Meta(abc.ABCMeta):
    def __getattr__(cls, name: str): 
        if name in cls._registered_dss:
            return cls._registered_dss[name]
        raise AttributeError(f"{cls.__name__} has no attribute {name}")

class Ds_Core(abc.ABC, metaclass=Ds_Core_Meta):
    """
    Класс, содержащий в себе все готовые решения для работы с датасетами.

    Attributes:
        _registered_dss: Словарь, содержащий все регистрированные решения

    Methods:
        __init_subclass__(cls, **kwargs): Регистрирует решение в словаре _registered_dss.
        get_registered_dss(): Возвращает словарь _registered_dss.
        get_all_dss(cls): Возвращает все решения, наследуемые от cls.
        ds(args: dict = {
            "original_bits": None,
            "extracted_bits": None
        }): Абстрактный метод решения, который должен быть реализован в каждом подклассе.
    """

    _registered_dss = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Ds_Core._registered_dss[cls.__name__] = cls

    def get_registered_dss():
        return Ds_Core._registered_dss

    @classmethod
    def get_all_dss(cls):
        return cls.__subclasses__()

    @staticmethod
    @abc.abstractmethod
    def ds(args: dict = {
        "original_bits": None,
        "extracted_bits": None
    }): 
        pass

    def get_ds_class_by_name(ds_name: str):
        return Ds_Core._registered_dss[ds_name]

    def use_dss(dss: dict):
        all_dss = dss.keys()
        for ds_name in all_dss:
            Ds_Core.get_ds_class_by_name(ds_name).ds(dss[ds_name])

class Ready_Datasets(Ds_Core):
    """
    Класс, содержащий в себе все готовые решения для работы с датасетами.
    """

# # Форматы, которые ядро считает валидными носителями ЦВЗ.
# SUPPORTED_EXTENSIONS: tuple = (".png", ".bmp", ".tiff", ".tif", ".ppm")


# class Ds_Core:
#     """
#     Оркестратор датасетов (dataset orchestrator).

#     Две задачи:
#     1. Регистрация сменных ds-решений
#     2. Поставка изображений-носителей в пайплайн
#     """

#     def __init__(self):
#         self.available_ds: dict = {}   # реестр подключённых ds-решений
#         self.dataset: list = []        # текущий список путей к изображениям
#         pass

#     def get_new_ds(self, ds_solutions: dict) -> None:
#         """
#         Регистрирует новые ds-решения, загружая их по пути к файлу.

#         Args:
#             ds_solutions (dict): отображение {имя: путь_к_py_файлу}, как в остальных Core.

#         Returns:
#             None
#         """
#         for ds in ds_solutions:
#             self.available_ds[ds] = import_function(ds, ds_solutions[ds])
#         return

#     def get_all_available_ds(self) -> None:
#         """
#         Печатает имена всех зарегистрированных ds-решений.

#         Returns:
#             None
#         """
#         for ds in self.available_ds:
#             print(ds)
#         return

#     def load_from_directory(self, directory: str, recursive: bool = False,
#                             extensions: tuple = SUPPORTED_EXTENSIONS) -> list:
#         """
#         Собирает список путей к изображениям-носителям из директории.

#         Args:
#             directory (str): путь к папке с изображениями.
#             recursive (bool): искать ли во вложенных папках.
#             extensions (tuple): допустимые расширения (по умолчанию без JPEG).

#         Returns:
#             list: отсортированный список абсолютных путей к изображениям.
#         """
#         root = Path(directory)
#         if not root.is_dir():
#             raise NotADirectoryError(f"Не директория: {directory}")
#         pattern = "**/*" if recursive else "*"
#         exts = {e.lower() for e in extensions}
#         self.dataset = sorted(
#             str(p.resolve())
#             for p in root.glob(pattern)
#             if p.is_file() and p.suffix.lower() in exts
#         )
#         return self.dataset

#     def __len__(self) -> int:
#         return len(self.dataset)

#     def __iter__(self) -> Iterator[str]:
#         """Итерация по путям текущего датасета"""
#         return iter(self.dataset)