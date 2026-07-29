from dwarf.common_utils.common_utils import *

_available = {module.name for module in pkgutil.iter_modules(__path__)
              if not module.ispkg}

for _name in sorted(_available):
    importlib.import_module(f"{__name__}.{_name}")

_uncompiled = sorted(
    source.stem for source in Path(__path__[0]).glob("*.pyx")
    if source.stem not in _available
)

if _uncompiled:
    warnings.warn(
        f"Не собраны Cython-расширения: {', '.join(_uncompiled)}. "
        f"Соответствующие метрики отсутствуют в реестре и не попадут в отчёт. "
        f"Соберите их командой: python setup.py build_ext --inplace",
        RuntimeWarning,
        stacklevel=2,
    )

del _available, _name, _uncompiled
