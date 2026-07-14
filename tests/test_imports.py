import importlib
import pkgutil

import dwarf
import dwarf.core


def test_public_api_imports():
    """Публичный API пакета импортируется (в т.ч. ds_core)."""
    from dwarf import AttackCore, EmbeddingCore, ExpertiseCore, ds_core


def test_core_modules_import():
    """Каждый модуль в dwarf.core импортируется.

    Ловит битые внутренние импорты (src.dwarf, кириллические гомоглифы).
    Обходим только core: там нет сторонних зависимостей, поэтому падение —
    это именно битый импорт, а не отсутствие numpy/PIL/torch.
    """
    failed = []
    for m in pkgutil.walk_packages(dwarf.core.__path__, dwarf.core.__name__ + "."):
        try:
            importlib.import_module(m.name)
        except Exception as e:
            failed.append((m.name, repr(e)))
    assert not failed, f"Не импортируются: {failed}"
