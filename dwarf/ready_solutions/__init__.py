"""Импорт пакета наполняет реестры ядра всеми готовыми решениями."""

import dwarf.ready_solutions.attack_solutions as attack_solutions
import dwarf.ready_solutions.ds_solutions as ds_solutions
import dwarf.ready_solutions.embedding_solutions as embedding_solutions
import dwarf.ready_solutions.expertise_solutions as expertise_solutions

__all__ = ["attack_solutions", "ds_solutions", "embedding_solutions", "expertise_solutions"]
