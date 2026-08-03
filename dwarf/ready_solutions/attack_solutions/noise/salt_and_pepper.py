from dwarf.ready_solutions.utils.attack_utils import *


class Salt_and_Pepper(Ready_Noise_Attacks):
    """
    Атака шумом типа "соль и перец".

    Заменяет случайную долю пикселей на чисто белые (соль) и чисто чёрные
    (перец) значения по всем каналам, поровну разделяя плотность между ними.
    """

    @staticmethod
    def attack(args: dict = {
        "input_image": [[[0]]],
        "density": 0.01,
        "seed": None
    }):
        """
        Наносит шум "соль и перец" на изображение и возвращает результат.

        Args:
            args (dict): параметры атаки
                input_image (list(list(list(int)))): матрица изображения
                density (float): суммарная доля повреждённых пикселей, диапазон [0.001, 0.05] (по умолчанию 0.01)
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

        height, width, _ = data.shape
        num_pixels = height * width

        num_salt = int(np.ceil(density * num_pixels * 0.5))
        num_pepper = int(np.ceil(density * num_pixels * 0.5))

        salt_rows = rng.integers(0, height, num_salt)
        salt_cols = rng.integers(0, width, num_salt)
        data[salt_rows, salt_cols, :] = 255

        pepper_rows = rng.integers(0, height, num_pepper)
        pepper_cols = rng.integers(0, width, num_pepper)
        data[pepper_rows, pepper_cols, :] = 0

        return data.tolist()