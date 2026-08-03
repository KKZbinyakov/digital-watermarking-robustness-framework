from dwarf.ready_solutions.utils.attack_utils import *


class AWGN(Ready_Noise_Attacks):
    """
    Атака аддитивным белым гауссовым шумом (AWGN).

    Добавляет к каждому пикселю независимый гауссов шум с нулевым средним,
    моделируя шум датчика или канала передачи.
    """

    @staticmethod
    def attack(args: dict = {
        "input_image": [[[0]]],
        "sigma": 0.01,
        "seed": None
    }):
        """
        Добавляет гауссов шум к изображению и возвращает результат.

        Args:
            args (dict): параметры атаки
                input_image (list(list(list(int)))): матрица изображения
                sigma (float): стандартное отклонение шума в долях канала 0..1, диапазон [0.001, 0.1] (по умолчанию 0.01)
                seed (int): зерно генератора случайных чисел (по умолчанию None)

        Returns:
            output_image (list(list(list(int)))): матрица изображения после атаки

        Raises:
            ValueError: если sigma вне диапазона [0.001, 0.1]
        """
        input_image = args["input_image"]
        sigma = float(args.get("sigma", 0.01))
        seed = args.get("seed", None)

        if not (0.001 <= sigma <= 0.1):
            raise ValueError("Sigma must be in range [0.001, 0.1]")

        rng = np.random.default_rng(seed)

        data = np.array(input_image, dtype=np.float32) / 255.0
        noise = rng.normal(0, sigma, data.shape)
        noisy_data = data + noise
        noisy_data = np.clip(noisy_data, 0, 1)
        return (noisy_data * 255).astype(np.uint8).tolist()