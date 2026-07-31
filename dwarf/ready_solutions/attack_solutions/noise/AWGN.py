from ...utils.attack_utils import *


class AWGN(Ready_Noise_Attacks):
    """
    Атака аддитивным белым гауссовым шумом (AWGN).

    Добавляет к каждому пикселю независимый гауссов шум с нулевым средним,
    моделируя шум датчика или канала передачи.
    """

    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Добавляет гауссов шум к изображению и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                sigma (float): стандартное отклонение шума в долях канала 0..1, диапазон [0.001, 0.1] (по умолчанию 0.01)
                seed (int): зерно генератора случайных чисел (по умолчанию None)

        Returns:
            None

        Raises:
            ValueError: если sigma вне диапазона [0.001, 0.1]
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        sigma = float(args.get("sigma", 0.01))
        seed = args.get("seed", None)
        
        if not (0.001 <= sigma <= 0.1):
            raise ValueError("Сигма должна быть в диапазоне [0.001, 0.1]")
        
        rng = np.random.default_rng(seed)
            
            
        img = Image.open(input_data).convert("RGB")
        data = np.array(img, dtype=np.float32) / 255.0
        noise = rng.normal(0, sigma, data.shape)
        noisy_data = data + noise
        noisy_data = np.clip(noisy_data, 0, 1)
        noisy_img = Image.fromarray((noisy_data * 255).astype(np.uint8))
        noisy_img.save(output_data)
        # print(f"Атака AWGN выполнена: сохранено {noisy_img.size} в {output_data}")