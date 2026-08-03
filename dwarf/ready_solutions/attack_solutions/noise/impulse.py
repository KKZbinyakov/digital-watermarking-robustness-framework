from dwarf.ready_solutions.utils.attack_utils import *


class Impulse(Ready_Noise_Attacks):
    """
    Атака импульсным шумом.

    Заменяет случайно выбранную долю пикселей на случайные значения по всем
    каналам, моделируя битые пиксели или сбои передачи.
    """

    @staticmethod
    def attack(args: dict = {
        "input_image": [[[0]]],
        "density": 0.01,
        "seed": None
    }):
        """
        Заменяет случайные пиксели изображения случайными значениями и возвращает результат.

        Args:
            args (dict): параметры атаки
                input_image (list(list(list(int)))): матрица изображения
                density (float): доля повреждённых пикселей, диапазон [0.001, 0.05] (по умолчанию 0.01)
                seed (int): зерно генератора случайных чисел (по умолчанию None)

        Returns:
            output_image (list(list(list(int)))): матрица изображения после атаки

        Raises:
            ValueError: если density вне диапазона [0.001, 0.05]
        """
        input_image = args["input_image"]
        density = float(args.get("density", 0.01))
        seed = args.get("seed", None)

        if not (0.001 <= density <= 0.05):
            raise ValueError("Density must be in range [0.001-0.05]")

        rng = np.random.default_rng(seed)

        data = np.array(input_image, dtype=np.uint8)

        height, width, channels = data.shape
        num_pixels = height * width
        num_impulses = int(np.ceil(density * num_pixels))

        rows = rng.integers(0, height, num_impulses)
        cols = rng.integers(0, width, num_impulses)
        values = rng.integers(0, 256, size=(num_impulses, channels), dtype=np.uint8)

        data[rows, cols, :] = values

        return data.tolist()