from dwarf.common_utils.common_utils import *

_available = {module.name
              for module in pkgutil.walk_packages(__path__, prefix=__name__ + ".")
              if not module.ispkg}

for _name in sorted(_available):
    importlib.import_module(_name)

_stems = {_full.rsplit(".", 1)[-1] for _full in _available}

_uncompiled = sorted(
    source.stem for source in Path(__path__[0]).rglob("*.pyx")
    if source.stem not in _stems
)

if _uncompiled:
    warnings.warn(
        f"Не собраны Cython-расширения: {', '.join(_uncompiled)}. "
        f"Соответствующие метрики отсутствуют в реестре и не попадут в отчёт. "
        f"Соберите их командой: python setup.py build_ext --inplace",
        RuntimeWarning,
        stacklevel=2,
    )

del _available, _name, _stems, _uncompiled
