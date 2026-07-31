from ...utils.attack_utils import *

class Box_Filter(Ready_Filtering_Attacks):
    """
    Атака прямоугольным (box) размытием.

    Заменяет каждый пиксель средним значением по квадратному окну соседних
    пикселей, равномерно размывая изображение и высокочастотный водяной знак.
    """

    @staticmethod
    def attack(args: dict = {
                "input_data": None,
                "output_data": None
    }):
        """
        Применяет прямоугольное размытие к изображению и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                window (int): размер окна усреднения, нечётное число >= 3 (по умолчанию 3)

        Returns:
            None

        Raises:
            ValueError: если window меньше 3 или чётное
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        window = args.get("window", 3)
        
        if window < 3 or window % 2 == 0:
            raise ValueError("Размер окна должен быть нечётным числом >= 3")
        
        img = Image.open(input_data).convert("RGB")
        filtered_img = img.filter(ImageFilter.BoxBlur(radius=window // 2))
        filtered_img.save(output_data)                  