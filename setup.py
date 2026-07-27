"""Сборка Cython-расширений пакета dwarf.

Метаданные проекта описаны в pyproject.toml, здесь задаётся только ext_modules
"""
import sys

from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

SRC = Path("src")

# Уровень оптимизации задаётся явно и по-разному для компиляторов: без него
# горячие циклы свёртки теряют около 15% скорости. Флаги, меняющие семантику
# вещественной арифметики, намеренно не используются — результат расширений
# должен побитово совпадать с реализацией на numpy.
OPTIMIZATION = ["/O2"] if sys.platform == "win32" else ["-O3"]

extensions = [
    Extension(
        name=".".join(pyx.relative_to(SRC).with_suffix("").parts),
        sources=[str(pyx)],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=OPTIMIZATION,
    )
    for pyx in sorted(SRC.rglob("*.pyx"))
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
