# Для теcтов
import src.dwаrf as dwarf

if __name__ == "__main__":
    Attack_Core = dwarf.AttackCore()
    Embedding_Core = dwarf.EmbeddingCore()
    Expertise_Core = dwarf.ExpertiseCore()

    Attack_Core.get_new_attack({"crop": r"C:\study\HSE\benchmark-project\src\dwаrf\ready_solutions\attack_solutions\crop.py"})
    Embedding_Core.get_new_embedding({"embed_and_extract":r"C:\study\HSE\benchmark-project\src\dwаrf\ready_solutions\embedding_solutions\embed_and_extract.py"})
    Expertise_Core.get_new_expertise({"ber":r"C:\study\HSE\benchmark-project\src\dwаrf\ready_solutions\expertise_solutions\ber.py"})

    Attack_Core.get_all_available_attacks()
    Embedding_Core.get_all_available_embeddings()
    Expertise_Core.get_all_available_expertise()

    Embedding_Core.available_embeddings["embed_and_extract"].embed_lsb("Рей.jpg", "1000110100000000000000000000000010001101000000000000000000000000", "embeded.jpg")
    Attack_Core.available_attacks["crop"].crop_attack("embeded.jpg", "croped.jpg")
    result = Embedding_Core.available_embeddings["embed_and_extract"].extract_lsb("croped.jpg", 32)
    print("BER:", Expertise_Core.available_expertise["ber"].compute_ber("10001101000000000000000000000000", result))

    pass