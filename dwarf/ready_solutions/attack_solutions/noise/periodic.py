from ...utils.attack_utils import *


class Periodic(Ready_Noise_Attacks):
    """
    Атака периодическим (синусоидальным) шумом.

    Накладывает на изображение синусоидальную волну вдоль горизонтальной оси
    со случайной фазой, моделируя помехи развёртки или растрирования.
    """

    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Накладывает синусоидальную помеху на изображение и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                amplitude (float): амплитуда волны в долях канала 0..1, диапазон [0.01, 0.2] (по умолчанию 0.05)
                frequency (float): частота волны по ширине изображения, диапазон [1.0, 100.0] (по умолчанию 10.0)
                seed (int): зерно генератора случайных чисел, определяет фазу волны (по умолчанию None)

        Returns:
            None

        Raises:
            ValueError: если amplitude вне диапазона [0.01, 0.2] или frequency вне диапазона [1.0, 100.0]
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        amplitude = float(args.get("amplitude", 0.05))
        frequency = float(args.get("frequency", 10.0))
        seed = args.get("seed", None)
        
        if not (0.01 <= amplitude <= 0.2):
            raise ValueError("Амплитуда должна быть в диапазоне [0.01-0.2]")
        if not (1.0 <= frequency <= 100.0):
            raise ValueError("Частота должна быть в диапазоне [1-100]")

        rng = np.random.default_rng(seed)

        img = Image.open(input_data).convert("RGB")
        data = np.array(img, dtype=np.float32) / 255.0

        height, width, _ = data.shape
        phase = rng.uniform(0.0, 2 * np.pi)

        x = np.arange(width, dtype=np.float32)
        wave = amplitude * np.sin(2 * np.pi * frequency * x / width + phase)

        noisy_data = data + wave[np.newaxis, :, np.newaxis]

        noisy_data = np.clip(noisy_data, 0, 1)
        noisy_img = Image.fromarray((noisy_data * 255).astype(np.uint8))
        noisy_img.save(output_data)
        # print(f"Атака Periodic noise выполнена (amplitude={amplitude}, frequency={frequency}): результат сохранён в {output_data}")