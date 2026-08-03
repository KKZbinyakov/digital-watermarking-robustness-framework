from dwarf.ready_solutions.utils.attack_utils import *
from PIL import ImageFilter

class Box_Filter(Ready_Filtering_Attacks):
    """
    Атака прямоугольным (box) размытием.

    Заменяет каждый пиксель средним значением по квадратному окну соседних
    пикселей, равномерно размывая изображение и высокочастотный водяной знак.
    """

    @staticmethod
    def attack(args: dict = {
                "input_image": [[[0]]],
                "window": 3
    }):
        """
        Применяет прямоугольное размытие к изображению и возвращает результат.

        Args:
            args (dict): параметры атаки
                input_image (list(list(list(int)))): матрица изображения
                window (int): размер окна усреднения, нечётное число >= 3 (по умолчанию 3)

        Returns:
            output_image (list(list(list(int)))): матрица изображения после атаки

        Raises:
            ValueError: если window меньше 3 или чётное
        """
        input_image = args["input_image"]
        window = args.get("window", 3)

        if window < 3 or window % 2 == 0:
            raise ValueError("Window size must be an odd number >= 3")

        img = Image.fromarray(np.array(input_image, dtype=np.uint8), "RGB")
        filtered_img = img.filter(ImageFilter.BoxBlur(radius=window // 2))
        return np.array(filtered_img).tolist()