# import sys
# from pathlib import Path
# PROJECT_DIR = Path(__file__).resolve().parent

# sys.path.insert(0, str(PROJECT_DIR))
# sys.path.insert(0, str(PROJECT_DIR / "src"))

from src.dwarf import *
# import src.dwarf as dwarf
import src.dwarf.ready_solutions as ready

from src.dwarf.ready_solutions.attack_solutions.crop import Crop
from src.dwarf.ready_solutions.embedding_solutions.lsb import LSB
from src.dwarf.ready_solutions.expertise_solutions.ber import BER
from src.dwarf.core.attack_orchestrator.attack_core import Ready_Geometric_Attacks

if __name__ == "__main__":
    exp_obj = Expertise_Core.BER
    emb_obj = Embedding_Core.LSB
    obj = Attack_Core.Crop

    print(emb_obj.__name__)
    print(obj.__name__)
    print(exp_obj.__name__)

    print(Attack_Core.get_all_attacks())

    Embedding_Core.LSB.embedding(args={"image_path": "Asuka.jpg", "watermark_bits": "1000110100000000000000000000000010001101000000000000000000000000", "output_path": "embeded.jpg"})
    Attack_Core.Crop.attack(args={"input_data": "embeded.jpg", "output_data": "croped.jpg"})
    Attack_Core.use_attacks({"Crop": {"input_data": "embeded.jpg", "output_data": "croped.jpg"}})
    result = Embedding_Core.LSB.extraction(args={"input_data": "croped.jpg", "num_bits": 32})
    print("BER:", Expertise_Core.BER.expertise(args={"original_bits": "10001101000000000000000000000000", "extracted_bits": result}))

    pass