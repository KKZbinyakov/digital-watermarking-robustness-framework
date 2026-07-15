# import sys
# from pathlib import Path
# PROJECT_DIR = Path(__file__).resolve().parent

# sys.path.insert(0, str(PROJECT_DIR))
# sys.path.insert(0, str(PROJECT_DIR / "src"))

import src.dwarf as dwarf
import src.dwarf.ready_solutions as ready

if __name__ == "__main__":
    Attack_Core = dwarf.Attack_Core()
    Embedding_Core = dwarf.Embedding_Core()
    Expertise_Core = dwarf.Expertise_Core()

    Attack_Core.connect_ready_attacks(connect_all=True)
    Embedding_Core.connect_ready_embeddings(connect_all=True)
    Expertise_Core.connect_ready_expertises(connect_all=True)

    # solutions_dir = PROJECT_DIR / "src" / "dwarf" / "ready_solutions"

    # Attack_Core.get_new_attack({
    #     "crop": str(solutions_dir / "attack_solutions" / "crop.py")
    # })
    # Embedding_Core.get_new_embedding({
    #     "embed_and_extract": str(solutions_dir / "embedding_solutions" / "embed_and_extract.py")
    # })
    # Expertise_Core.get_new_expertise({
    #     "ber": str(solutions_dir / "expertise_solutions" / "ber.py")
    # })

    Attack_Core.get_all_available_attacks()
    Embedding_Core.get_all_available_embeddings()
    # print(Embedding_Core.available_embeddings["spatial_embeddings"])
    Expertise_Core.get_all_available_expertise()

    Embedding_Core.spatial_embeddings.Ready_Spatial_Embeddings.embed_lsb("Asuka.jpg", "1000110100000000000000000000000010001101000000000000000000000000", "embeded.jpg")
    # Attack_Core.available_attacks["geometric_attacks"].Ready_Geometric_Attacks.crop_attack("embeded.jpg", "croped.jpg")
    Attack_Core.geometric_attacks.Ready_Geometric_Attacks.crop_attack("embeded.jpg", "croped.jpg")
    result = Embedding_Core.spatial_embeddings.Ready_Spatial_Embeddings.extract_lsb("croped.jpg", 32)
    print("BER:", Expertise_Core.robustness_expertises.Ready_Robustness_Expertise.ber("10001101000000000000000000000000", result))

    pass