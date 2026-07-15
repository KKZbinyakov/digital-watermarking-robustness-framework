import numpy as np

from pathlib import Path
from PIL import Image
from src.dwarf.common_utils.common_utils import *
# def scan_files_in_dir(dir_path: str) -> list:
#     result = []

#     files = dir_path.iterdir()

#     for file in files:
#         if file.is_file():
#             file_name = file.name
#             if file_name == "__init__.py" or file_name == "__pycache__":
#                 continue
#             result.append(file_name)

#     return result