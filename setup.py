from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

ext_modules = cythonize(
    [
        "dwarf/ready_solutions/utils/embedding_utils_pyx.pyx",
        "dwarf/ready_solutions/embedding_solutions/frequency/contourlet.pyx",
    ],
    compiler_directives={
        'language_level': "3",
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True
    },
    force=True,
)

setup(
    name="dwarf",
    packages=find_packages(),
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
)