import importlib.util

from src.dwarf.common_utils.common_utils import *

def import_function(module_name: str, file_path: str):
    print(f"Импорт модуля {module_name} из {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module) 
    return module
    