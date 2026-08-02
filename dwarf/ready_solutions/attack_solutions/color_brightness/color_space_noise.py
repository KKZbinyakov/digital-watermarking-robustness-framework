from ...utils.attack_utils import *


class Color_Space_Noise(Ready_Color_Brightness_Attacks):
    """
    Атака гауссовым шумом в заданном цветовом пространстве.
    """

    @staticmethod
    def attack(args: dict = {"input_data": None, "output_data": None}):
        """
        Добавляет шум в заданном цветовом пространстве и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                space (str): цветовое пространство Pillow, 'YCbCr', 'HSV' или 'LAB' (по умолчанию 'YCbCr')
                noise_std (float): стандартное отклонение шума в единицах канала 0..255 (по умолчанию 10.0)
                seed (int): зерно генератора случайных чисел (по умолчанию None)

        Returns:
            None
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        space = args.get("space", "YCbCr")
        noise_std = float(args.get("noise_std", 10.0))
        seed = args.get("seed")

        rng = np.random.default_rng(seed)
        img = Image.open(input_data).convert("RGB")
        converted = np.asarray(img.convert(space)).astype(np.float64)
        converted += rng.normal(0.0, noise_std, converted.shape)
        converted = np.clip(converted, 0, 255).round().astype(np.uint8)
        Image.fromarray(converted, space).convert("RGB").save(output_data)
