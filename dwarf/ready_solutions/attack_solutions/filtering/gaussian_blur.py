from ...utils.attack_utils import *

class Gaussian_Blur(Ready_Filtering_Attacks):
    """
    Атака гауссовым размытием.

    Свёртывает изображение с гауссовым ядром, подавляя высокочастотные
    детали, в которых обычно скрыт водяной знак.
    """

    @staticmethod
    def attack(args: dict = {
            "input_data": None,
            "output_data": None
    }):
        """
        Применяет гауссово размытие к изображению и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                sigma (float): радиус (стандартное отклонение) размытия, диапазон [0.5, 5.0] (по умолчанию 1.0)

        Returns:
            None

        Raises:
            ValueError: если sigma вне диапазона [0.5, 5.0]
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        sigma = float(args.get("sigma", 1.0))
        
        if not (0.5 <= sigma <= 5.0):
            raise ValueError("Sigma должна быть в диапазоне [0.5-5.0]")
        
        img = Image.open(input_data).convert("RGB")
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        blurred_img.save(output_data)