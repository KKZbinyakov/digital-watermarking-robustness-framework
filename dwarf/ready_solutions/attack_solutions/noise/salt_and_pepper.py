from ...utils.attack_utils import *


class Salt_and_Pepper(Ready_Noise_Attacks):
    """
    Атака шумом типа "соль и перец".

    Заменяет случайную долю пикселей на чисто белые (соль) и чисто чёрные
    (перец) значения по всем каналам, поровну разделяя плотность между ними.
    """

    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Наносит шум "соль и перец" на изображение и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                density (float): суммарная доля повреждённых пикселей, диапазон [0.001, 0.05] (по умолчанию 0.01)
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

        noisy_img = Image.fromarray(data)
        noisy_img.save(output_data)
        # print(f"Атака Salt & Pepper выполнена: сохранено {noisy_img.size} в {output_data}")        