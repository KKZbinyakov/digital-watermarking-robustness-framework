from dwarf.ready_solutions.utils.attack_utils import *


class Speckle(Ready_Noise_Attacks):
    """
    Атака мультипликативным (спекл) шумом.

    Добавляет к пикселю шум, пропорциональный его собственной яркости
    (pixel + pixel * noise), моделируя спекл-шум когерентных изображающих систем.
    """

    @staticmethod
    def attack(args: dict = {
        "input_image": [[[0]]],
        "variance": 0.05,
        "seed": None
    }):
        """
        Добавляет мультипликативный шум к изображению и возвращает результат.

        Args:
            args (dict): параметры атаки
                input_image (list(list(list(int)))): матрица изображения
                variance (float): дисперсия шума, диапазон [0.001, 0.05] (по умолчанию 0.05)
                seed (int): зерно генератора случайных чисел (по умолчанию None)

        Returns:
            output_image (list(list(list(int)))): матрица изображения после атаки

        Raises:
            ValueError: если variance вне диапазона [0.001, 0.05]
        """
        input_image = args["input_image"]
        variance = float(args.get("variance", 0.05))
        seed = args.get("seed", None)

        if not (0.001 <= variance <= 0.05):
            raise ValueError("Variance must be in range [0.001-0.05]")

        rng = np.random.default_rng(seed)

        data = np.array(input_image, dtype=np.float32) / 255.0

        sigma = np.sqrt(variance)
        noise = rng.normal(0.0, sigma, data.shape).astype(np.float32)

        noisy_data = data + data * noise

        noisy_data = np.clip(noisy_data, 0, 1)
        return (noisy_data * 255).astype(np.uint8).tolist()