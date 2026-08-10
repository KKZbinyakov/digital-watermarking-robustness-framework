# Определяет директорию, в которой находится python-пакетом. Здесь можно написать код, который должен выполняться при импорте пакета.
from .core import Attack_Core, Ds_Core, Embedding_Core, Expertise_Core

__all__ = ["Attack_Core", "Embedding_Core", "Expertise_Core", "Ds_Core"] # Контроль импорта, определяет, котое имена из этого модуля должны быть импортированы. Таким образом для импорта ready_solutions нужен явный импорт.
