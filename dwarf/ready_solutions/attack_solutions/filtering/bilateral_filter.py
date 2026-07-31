from ...utils.attack_utils import *

class Bilateral_Filter(Ready_Filtering_Attacks):
    """
    Атака билатеральной фильтрацией.

    Сглаживает изображение с учётом как пространственной близости, так и
    близости по цвету, что позволяет размывать шум и водяной знак, сохраняя
    резкие границы.
    """

    @staticmethod
    def attack(args: dict = {
            "input_data": None,
            "output_data": None
    }):
        """
        Применяет билатеральный фильтр к изображению и сохраняет результат.

        Диаметр окна фильтра вычисляется из sigma_space, чтобы захватывать
        значимую часть гауссова пространственного ядра.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                sigma_color (float): сила сглаживания по цвету в долях канала 0..1, диапазон [0.01, 0.5] (по умолчанию 0.1)
                sigma_space (float): сила пространственного сглаживания в пикселях, диапазон [1.0, 10.0] (по умолчанию 3.0)

        Returns:
            None

        Raises:
            ValueError: если sigma_color вне диапазона [0.01, 0.5] или sigma_space вне диапазона [1.0, 10.0]
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        sigma_color = float(args.get("sigma_color", 0.1))
        sigma_space = float(args.get("sigma_space", 3.0))
        
        if not (0.01 <= sigma_color <= 0.5):
            raise ValueError("sigma_color должна быть в диапазоне [0.01-0.5]")
        if not (1.0 <= sigma_space <= 10.0):
            raise ValueError("sigma_space должна быть в диапазоне [1.0-10.0]")
        
        img = Image.open(input_data).convert("RGB")
        data = np.array(img, dtype=np.float32) / 255.0
        
        d = int(2 * np.ceil(2 * sigma_space) + 1) # Диаметр окна
        
        filtered = cv2.bilateralFilter(data, d=d,
                                    sigmaColor=sigma_color,
                                    sigmaSpace=sigma_space)

        filtered = np.clip(filtered, 0, 1)
        filtered_img = Image.fromarray((filtered * 255).astype(np.uint8))
        filtered_img.save(output_data)
        
        