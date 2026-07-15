# Определяет директорию, в которой находится python-пакетом. Здесь можно написать код, который должен выполняться при импорте пакета.
# from core.ds_orchestrator import DsCore
from .core.utils.utils import *
from .core.attack_orchestrator.attack_core import Attack_Core
from .core.embedding_orchestrator.embedding_core import Embedding_Core
from .core.expertise_orchestrator.expertise_core import Expertise_Core
from .core.ds_orchestrator.ds_core import Ds_Core
from .core.__init__ import *

print("CORE_DIR:", CORE_DIR)
print("DWARF_DIR:", DWARF_DIR)

# DWARF_DIR = Path(__file__).resolve().parent

__all__ = ["Attack_Core", "Embedding_Core", "Expertise_Core", "Ds_Core"] # Контроль импорта, определяет, котое имена из этого модуля должны быть импортированы. Таким образом для импорта ready_solutions нужен явный импорт.