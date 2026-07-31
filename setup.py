"""Сборка Cython-расширений пакета dwarf.

Метаданные проекта описаны в pyproject.toml, здесь задаётся только ext_modules
"""
import os
import sys

from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

SRC = Path(".")
PACKAGE = Path("dwarf")

# Уровень оптимизации задаётся явно и по-разному для компиляторов: без него
# горячие циклы свёртки теряют около 15% скорости. Флаги, меняющие семантику
# вещественной арифметики, намеренно не используются — результат расширений
# должен побитово совпадать с реализацией на numpy.
OPTIMIZATION = ["/O2"] if sys.platform == "win32" else ["-O3"]

# Расширения, использующие prange. Без флагов OpenMP Cython генерирует код под
# #ifdef _OPENMP, поэтому сборка не падает, а молча получается последовательной.
# Отключить OpenMP осознанно (например, на macOS без libomp) можно переменной
# окружения DWARF_NO_OPENMP=1.
PARALLEL = {"dwarf.ready_solutions.embedding_solutions.frequency.svd"}

OPENMP_COMPILE = ["/openmp"] if sys.platform == "win32" else ["-fopenmp"]
OPENMP_LINK = [] if sys.platform == "win32" else ["-fopenmp"]

USE_OPENMP = os.environ.get("DWARF_NO_OPENMP", "") != "1"


def build_extension(pyx):
    """
    Собирает описание расширения для одного .pyx.

    Args:
        pyx (Path): путь к исходнику относительно корня репозитория

    Returns:
        Extension: описание расширения для setuptools
    """
    name = ".".join(pyx.relative_to(SRC).with_suffix("").parts)
    parallel = USE_OPENMP and name in PARALLEL
    return Extension(
        name=name,
        sources=[str(pyx)],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=OPTIMIZATION + (OPENMP_COMPILE if parallel else []),
        extra_link_args=OPENMP_LINK if parallel else [],
    )


extensions = [build_extension(pyx) for pyx in sorted(PACKAGE.rglob("*.pyx"))]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
