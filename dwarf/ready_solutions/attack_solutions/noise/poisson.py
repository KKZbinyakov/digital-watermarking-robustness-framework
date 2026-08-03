from dwarf.ready_solutions.utils.attack_utils import *


class Poisson(Ready_Noise_Attacks):
    """
    Атака пуассоновским (дробовым) шумом.

    Моделирует шум фотонного дробового эффекта: масштабирует яркость пикселя
    к среднему числу фотонов peak, сэмплирует пуассоновскую случайную величину
    и масштабирует обратно.
    """

    @staticmethod
    def attack(args: dict = {
        "input_image": [[[0]]],
        "peak": 30.0,
        "seed": None
    }):
        """
        Добавляет пуассоновский шум к изображению и возвращает результат.

        Args:
            args (dict): параметры атаки
                input_image (list(list(list(int)))): матрица изображения
                peak (float): среднее число фотонов на максимум яркости, диапазон [1.0, 1000.0] (по умолчанию 30.0)
                seed (int): зерно генератора случайных чисел (по умолчанию None)

        Returns:
            output_image (list(list(list(int)))): матрица изображения после атаки

        Raises:
            ValueError: если peak вне диапазона [1.0, 1000.0]
        """
        input_image = args["input_image"]
        peak = float(args.get("peak", 30.0))
        seed = args.get("seed", None)

        if not (1.0 <= peak <= 1000.0):
            raise ValueError("Parameter peak must be in range [1-1000]")

        rng = np.random.default_rng(seed)

        data = np.array(input_image, dtype=np.float32) / 255.0

        scaled = data * peak

        noisy_data = rng.poisson(scaled).astype(np.float32) / peak

        noisy_data = np.clip(noisy_data, 0, 1)
        return (noisy_data * 255).astype(np.uint8).tolist()