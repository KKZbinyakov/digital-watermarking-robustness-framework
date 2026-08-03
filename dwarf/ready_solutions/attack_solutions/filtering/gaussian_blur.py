from dwarf.ready_solutions.utils.attack_utils import *
from PIL import ImageFilter

class Gaussian_Blur(Ready_Filtering_Attacks):
    """
    Атака гауссовым размытием.

    Свёртывает изображение с гауссовым ядром, подавляя высокочастотные
    детали, в которых обычно скрыт водяной знак.
    """

    @staticmethod
    def attack(args: dict = {
            "input_image": [[[0]]],
            "sigma": 1.0
    }):
        """
        Применяет гауссово размытие к изображению и возвращает результат.

        Args:
            args (dict): параметры атаки
                input_image (list(list(list(int)))): матрица изображения
                sigma (float): радиус (стандартное отклонение) размытия, диапазон [0.5, 5.0] (по умолчанию 1.0)

        Returns:
            output_image (list(list(list(int)))): матрица изображения после атаки

        Raises:
            ValueError: если sigma вне диапазона [0.5, 5.0]
        """
        input_image = args["input_image"]
        sigma = float(args.get("sigma", 1.0))

        if not (0.5 <= sigma <= 5.0):
            raise ValueError("Sigma must be in range [0.5-5.0]")

        img = Image.fromarray(np.array(input_image, dtype=np.uint8), "RGB")
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.array(blurred_img).tolist()