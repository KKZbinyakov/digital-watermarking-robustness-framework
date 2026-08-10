"""Папка реализации должна совпадать с её категорией из базового класса."""

import inspect

import pytest
from conftest import solutions

import dwarf.ready_solutions  # noqa: F401  наполняет реестры всех категорий
from dwarf.core.attack_orchestrator.attack_core import Attack_Core
from dwarf.core.expertise_orchestrator.expertise_core import Expertise_Core

SUFFIXES = ("_Attacks", "_Expertise", "_Embeddings", "_Datasets")


def category_folder(base_name):
    """
    Выводит имя папки из имени категории.

    Args:
        base_name (str): имя базового класса вида Ready_Compression_Attacks

    Returns:
        str: ожидаемое имя папки
    """
    name = base_name[len("Ready_") :]
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)].lower()
    return name.lower()


REGISTERED = list(solutions(Attack_Core.get_registered_attacks()).values()) + list(
    solutions(Expertise_Core.get_registered_expertises()).values()
)


@pytest.mark.parametrize("cls", REGISTERED, ids=lambda cls: cls.__name__)
def test_folder_matches_category(cls):
    bases = [base.__name__ for base in cls.__mro__[1:] if base.__name__.startswith("Ready_")]
    assert bases, f"{cls.__name__} не наследует ни одну категорию Ready_*"

    expected = category_folder(bases[0])
    path = inspect.getfile(cls).replace("\\", "/")
    assert f"/{expected}/" in path, f"{cls.__name__} наследует {bases[0]} (категория {expected!r}), но лежит в {path}"


def test_every_registered_solution_has_a_category():
    assert REGISTERED, "реестры пусты: решения не импортировались"
