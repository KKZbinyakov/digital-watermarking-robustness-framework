"""Папка реализации должна совпадать с её категорией из базового класса."""
import inspect
from pathlib import Path
import pytest
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dwarf.core.attack_orchestrator.attack_core import Attack_Core
from dwarf.core.expertise_orchestrator.expertise_core import Expertise_Core

SUFFIXES = ("_Attacks", "_Expertise", "_Embeddings", "_Datasets")


def category_folder(base_name):
    name = base_name[len("Ready_"):]
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)].lower()
    return name.lower()


def implementations(registry):
    """Только конкретные реализации: сами категории Ready_* пропускаем."""
    return [cls for name, cls in sorted(registry.items())
            if not name.startswith("Ready_")]


@pytest.mark.parametrize(
    "cls",
    implementations(Attack_Core.get_registered_attacks())
    + implementations(Expertise_Core.get_registered_expertises()),
    ids=lambda cls: cls.__name__,
)
def test_folder_matches_category(cls):
    bases = [b.__name__ for b in cls.__mro__[1:] if b.__name__.startswith("Ready_")]
    assert bases, f"{cls.__name__} не наследует ни одну категорию Ready_*"

    expected = category_folder(bases[0])
    path = inspect.getfile(cls).replace("\\", "/")
    assert f"/{expected}/" in path, (
        f"{cls.__name__} наследует {bases[0]} (категория {expected!r}), "
        f"но лежит в {path}"
    )
