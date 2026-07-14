# Определяет директорию, в которой находится python-пакетом. Здесь можно написать код, который должен выполняться при импорте пакета.
# from core.ds_orchestrator import DsCore
from .core.utils.utils import *
from .core.attack_orchestrator.attack_core import AttackCore
from .core.embedding_orchestrator.embedding_core import EmbeddingCore
from .core.expertise_orchestrator.expertise_core import ExpertiseCore
from .core.ds_orchestrator.ds_core import ds_core

__all__ = ["AttackCore", "EmbeddingCore", "ExpertiseCore", "ds_core"] # Контроль импорта, определяет, котое имена из этого модуля должны быть импортированы. Таким образом для импорта ready_solutions нужен явный импорт.