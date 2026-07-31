from ...utils.attack_utils import *


class Speckle(Ready_Noise_Attacks):
    """
    Атака мультипликативным (спекл) шумом.

    Добавляет к пикселю шум, пропорциональный его собственной яркости
    (pixel + pixel * noise), моделируя спекл-шум когерентных изображающих систем.
    """

    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Добавляет мультипликативный шум к изображению и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                variance (float): дисперсия шума, диапазон [0.001, 0.05] (по умолчанию 0.05)
                seed (int): зерно генератора случайных чисел (по умолчанию None)

        Returns:
            None

        Raises:
            ValueError: если variance вне диапазона [0.001, 0.05]
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        variance = float(args.get("variance", 0.05))
        seed = args.get("seed", None)
        
        if not (0.001 <= variance <= 0.05):
            raise ValueError("Дисперсия должна быть в диапазоне [0.001-0.05]")
        
        rng = np.random.default_rng(seed)
        
        img = Image.open(input_data).convert("RGB")
        data = np.array(img, dtype=np.float32) / 255.0
        
        sigma = np.sqrt(variance)
        noise = rng.normal(0.0, sigma, data.shape).astype(np.float32)
        
        noisy_data = data + data * noise
        
        noisy_data = np.clip(noisy_data, 0, 1)
        noisy_img = Image.fromarray((noisy_data * 255).astype(np.uint8))
        noisy_img.save(output_data)
        # print(f"Атака Speckle выполнена (variance={variance}): результат сохранён в {output_data}")  
        
        