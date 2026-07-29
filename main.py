from dwarf import *

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