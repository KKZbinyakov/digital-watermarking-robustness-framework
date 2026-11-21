"""Build only the Cython extensions used by the DCT, DWT and DFT paper runs."""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(output.parent)
    source_root = Path("cython_sources")
    source_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "dwarf.ready_solutions.utils.embedding_utils_pyx": Path("dwarf/ready_solutions/utils/embedding_utils_pyx.pyx"),
        **{
            f"dwarf.ready_solutions.embedding_solutions.frequency.{name}": Path(
                f"dwarf/ready_solutions/embedding_solutions/frequency/{name}.pyx"
            )
            for name in ("dct", "dwt", "dft")
        },
    }
    pxd = Path("dwarf/ready_solutions/utils/embedding_utils_pyx.pxd")
    for name, relative in paths.items():
        destination = source_root / f"{name.rsplit('.', 1)[-1]}.pyx"
        shutil.copyfile(repo / relative, destination)
    for destination in (source_root / pxd, source_root / pxd.name):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / pxd, destination)

    optimization = ["/O2"] if sys.platform == "win32" else ["-O3"]
    extensions = [
        Extension(
            name,
            [str(source_root / f"{name.rsplit('.', 1)[-1]}.pyx")],
            include_dirs=[numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=optimization,
        )
        for name, relative in paths.items()
    ]
    setup(
        name="frequency_paper_test_backend",
        ext_modules=cythonize(
            extensions,
            build_dir="cython_c",
            include_path=[str(source_root)],
            compiler_directives={"language_level": "3"},
            force=True,
        ),
        script_args=[
            "build_ext",
            "--build-lib",
            str(output),
            "--build-temp",
            "objects",
        ],
    )


if __name__ == "__main__":
    main()
