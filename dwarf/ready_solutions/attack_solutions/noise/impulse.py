from ...utils.attack_utils import *


class Impulse(Ready_Noise_Attacks):
    """
    Атака импульсным шумом.

    Заменяет случайно выбранную долю пикселей на случайные значения по всем
    каналам, моделируя битые пиксели или сбои передачи.
    """

    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Заменяет случайные пиксели изображения случайными значениями и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                density (float): доля повреждённых пикселей, диапазон [0.001, 0.05] (по умолчанию 0.01)
                seed (int): зерно генератора случайных чисел (по умолчанию None)

        Returns:
            None

        Raises:
            ValueError: если density вне диапазона [0.001, 0.05]
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        density = float(args.get("density", 0.01))
        seed = args.get("seed", None)
        
        if not (0.001 <= density <= 0.05):
            raise ValueError("Плотность должна быть в диапазоне [0.001-0.05]")

        rng = np.random.default_rng(seed)

        img = Image.open(input_data).convert("RGB")
        data = np.array(img, dtype=np.uint8)

        height, width, channels = data.shape
        num_pixels = height * width
        num_impulses = int(np.ceil(density * num_pixels))

        rows = rng.integers(0, height, num_impulses)
        cols = rng.integers(0, width, num_impulses)
        values = rng.integers(0, 256, size=(num_impulses, channels), dtype=np.uint8)

        data[rows, cols, :] = values

        noisy_img = Image.fromarray(data)
        noisy_img.save(output_data)
        # print(f"Атака Impulse noise выполнена (density={density}): результат сохранён в {output_data}")      