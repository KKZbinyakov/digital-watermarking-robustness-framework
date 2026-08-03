from dwarf.ready_solutions.utils.attack_utils import *


class Periodic(Ready_Noise_Attacks):
    """
    Атака периодическим (синусоидальным) шумом.

    Накладывает на изображение синусоидальную волну вдоль горизонтальной оси
    со случайной фазой, моделируя помехи развёртки или растрирования.
    """

    @staticmethod
    def attack(args: dict = {
        "input_image": [[[0]]],
        "amplitude": 0.05,
        "frequency": 10.0,
        "seed": None
    }):
        """
        Накладывает синусоидальную помеху на изображение и возвращает результат.

        Args:
            args (dict): параметры атаки
                input_image (list(list(list(int)))): матрица изображения
                amplitude (float): амплитуда волны в долях канала 0..1, диапазон [0.01, 0.2] (по умолчанию 0.05)
                frequency (float): частота волны по ширине изображения, диапазон [1.0, 100.0] (по умолчанию 10.0)
                seed (int): зерно генератора случайных чисел, определяет фазу волны (по умолчанию None)

        Returns:
            output_image (list(list(list(int)))): матрица изображения после атаки

        Raises:
            ValueError: если amplitude вне диапазона [0.01, 0.2] или frequency вне диапазона [1.0, 100.0]
        """
        input_image = args["input_image"]
        amplitude = float(args.get("amplitude", 0.05))
        frequency = float(args.get("frequency", 10.0))
        seed = args.get("seed", None)

        if not (0.01 <= amplitude <= 0.2):
            raise ValueError("Amplitude must be in range [0.01-0.2]")
        if not (1.0 <= frequency <= 100.0):
            raise ValueError("Frequency must be in range [1-100]")

        rng = np.random.default_rng(seed)

        data = np.array(input_image, dtype=np.float32) / 255.0

        height, width, _ = data.shape
        phase = rng.uniform(0.0, 2 * np.pi)

        x = np.arange(width, dtype=np.float32)
        wave = amplitude * np.sin(2 * np.pi * frequency * x / width + phase)

        noisy_data = data + wave[np.newaxis, :, np.newaxis]

        noisy_data = np.clip(noisy_data, 0, 1)
        return (noisy_data * 255).astype(np.uint8).tolist()