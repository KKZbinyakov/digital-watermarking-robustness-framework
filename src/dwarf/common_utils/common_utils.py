import sys
import abc

from pathlib import Path
from typing import TYPE_CHECKING, Any


DWARF_DIR = Path(__file__).resolve().parents[1]

# PROJECT_DIR = Path(__file__).resolve().parent

# sys.path.insert(0, str(PROJECT_DIR))
# sys.path.insert(0, str(PROJECT_DIR / "src"))