from ...utils.attack_utils import *


class Color_Jitter(Ready_Color_Brightness_Attacks):
    """
    Атака случайного цветового дрожания.

    Случайно меняет яркость, контраст, насыщенность и оттенок в пределах
    заданного размаха, моделируя нестабильную цветокоррекцию при публикации
    через разные сервисы. Каждый параметр задаёт полуширину равномерного
    отклонения от исходного значения.
    """

    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Применяет случайное цветовое дрожание и сохраняет результат.

        Оттенок сдвигается в пространстве HSV, где канал H занимает диапазон
        0..255 и замкнут по кругу, поэтому сдвиг берётся по модулю 256.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                brightness (float): размах яркости, множитель 1 +- brightness (по умолчанию 0.2)
                contrast (float): размах контраста, множитель 1 +- contrast (по умолчанию 0.2)
                saturation (float): размах насыщенности, множитель 1 +- saturation (по умолчанию 0.2)
                hue (float): размах сдвига оттенка в долях цветового круга (по умолчанию 0.05)
                seed (int): зерно генератора случайных чисел, без него атака невоспроизводима (по умолчанию None)

        Returns:
            None
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        brightness = float(args.get("brightness", 0.2))
        contrast = float(args.get("contrast", 0.2))
        saturation = float(args.get("saturation", 0.2))
        hue = float(args.get("hue", 0.05))
        seed = args.get("seed", None)

        rng = np.random.default_rng(seed)
        img = Image.open(input_data).convert("RGB")
        img = ImageEnhance.Brightness(img).enhance(1 + rng.uniform(-brightness, brightness))
        img = ImageEnhance.Contrast(img).enhance(1 + rng.uniform(-contrast, contrast))
        img = ImageEnhance.Color(img).enhance(1 + rng.uniform(-saturation, saturation))

        hsv = np.asarray(img.convert("HSV")).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.uniform(-hue, hue) * 255)) % 256
        Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB").save(output_data)
