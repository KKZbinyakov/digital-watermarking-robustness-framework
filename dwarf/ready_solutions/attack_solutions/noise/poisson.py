from ...utils.attack_utils import *


class Poisson(Ready_Noise_Attacks):
    """
    Атака пуассоновским (дробовым) шумом.

    Моделирует шум фотонного дробового эффекта: масштабирует яркость пикселя
    к среднему числу фотонов peak, сэмплирует пуассоновскую случайную величину
    и масштабирует обратно.
    """

    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Добавляет пуассоновский шум к изображению и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                peak (float): среднее число фотонов на максимум яркости, диапазон [1.0, 1000.0] (по умолчанию 30.0)
                seed (int): зерно генератора случайных чисел (по умолчанию None)

        Returns:
            None

        Raises:
            ValueError: если peak вне диапазона [1.0, 1000.0]
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        peak = float(args.get("peak", 30.0))
        seed = args.get("seed", None)

        if not (1.0 <= peak <= 1000.0):
            raise ValueError("Параметр peak должен быть в диапазоне [1-1000]")

        rng = np.random.default_rng(seed)

        img = Image.open(input_data).convert("RGB")
        data = np.array(img, dtype=np.float32) / 255.0

        scaled = data * peak

        noisy_data = rng.poisson(scaled).astype(np.float32) / peak

        noisy_data = np.clip(noisy_data, 0, 1)
        noisy_img = Image.fromarray((noisy_data * 255).astype(np.uint8))
        noisy_img.save(output_data)
        # print(f"Атака Poisson выполнена (peak={peak}): результат сохранён в {output_data}")    