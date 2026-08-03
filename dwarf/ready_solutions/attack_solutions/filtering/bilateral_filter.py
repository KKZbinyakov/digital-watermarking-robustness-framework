from dwarf.ready_solutions.utils.attack_utils import *
import cv2

class Bilateral_Filter(Ready_Filtering_Attacks):
    """
    Атака билатеральной фильтрацией.

    Сглаживает изображение с учётом как пространственной близости, так и
    близости по цвету, что позволяет размывать шум и водяной знак, сохраняя
    резкие границы.
    """

    @staticmethod
    def attack(args: dict = {
            "input_image": [[[0]]],
            "sigma_color": 0.1,
            "sigma_space": 3.0
    }):
        """
        Применяет билатеральный фильтр к изображению и возвращает результат.

        Диаметр окна фильтра вычисляется из sigma_space, чтобы захватывать
        значимую часть гауссова пространственного ядра.

        Args:
            args (dict): параметры атаки
                input_image (list(list(list(int)))): матрица изображения
                sigma_color (float): сила сглаживания по цвету в долях канала 0..1, диапазон [0.01, 0.5] (по умолчанию 0.1)
                sigma_space (float): сила пространственного сглаживания в пикселях, диапазон [1.0, 10.0] (по умолчанию 3.0)

        Returns:
            output_image (list(list(list(int)))): матрица изображения после атаки

        Raises:
            ValueError: если sigma_color вне диапазона [0.01, 0.5] или sigma_space вне диапазона [1.0, 10.0]
        """
        input_image = args["input_image"]
        sigma_color = float(args.get("sigma_color", 0.1))
        sigma_space = float(args.get("sigma_space", 3.0))

        if not (0.01 <= sigma_color <= 0.5):
            raise ValueError("sigma_color must be in range [0.01-0.5]")
        if not (1.0 <= sigma_space <= 10.0):
            raise ValueError("sigma_space must be in range [1.0-10.0]")

        data = np.array(input_image, dtype=np.float32) / 255.0

        d = int(2 * np.ceil(2 * sigma_space) + 1) # Диаметр окна

        filtered = cv2.bilateralFilter(data, d=d,
                                    sigmaColor=sigma_color,
                                    sigmaSpace=sigma_space)

        filtered = np.clip(filtered, 0, 1)
        return (filtered * 255).astype(np.uint8).tolist()
