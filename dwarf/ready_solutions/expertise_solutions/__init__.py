"""Готовые реализации метрик экспертизы.

Классы метрик регистрируются в Expertise_Core через __init_subclass__, то есть
только как побочный эффект импорта своего модуля. Поэтому пакет импортирует все
свои модули при загрузке: без этого Expertise_Core.get_registered_expertises()
вернёт одни лишь категории, а обращение вида Expertise_Core.SSIM поднимет
AttributeError.

Несобранные .pyx для pkgutil невидимы, поэтому их отсутствие не ломает импорт,
а лишь выкидывает соответствующие метрики из реестра. В прогоне бенчмарка это
опаснее ошибки импорта: отчёт получится неполным, и понять это по нему нельзя.
Такой случай разбирается отдельно и сопровождается предупреждением.
"""

from dwarf.common_utils.common_utils import *

_available = {module.name for module in pkgutil.walk_packages(__path__, prefix=__name__ + ".") if not module.ispkg}

for _name in sorted(_available):
    importlib.import_module(_name)

_stems = {_full.rsplit(".", 1)[-1] for _full in _available}

_uncompiled = sorted(source.stem for source in Path(__path__[0]).rglob("*.pyx") if source.stem not in _stems)

if _uncompiled:
    warnings.warn(
        f"Не собраны Cython-расширения: {', '.join(_uncompiled)}. "
        f"Соответствующие метрики отсутствуют в реестре и не попадут в отчёт. "
        f"Соберите их командой: python setup.py build_ext --inplace",
        RuntimeWarning,
        stacklevel=2,
    )

del _available, _name, _stems, _uncompiled
