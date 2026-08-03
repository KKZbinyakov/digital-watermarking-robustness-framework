from dwarf.ready_solutions.utils.attack_utils import *
from PIL import ImageFilter

class Unsharp_Mask(Ready_Filtering_Attacks):
    """
    Атака нерезким маскированием (unsharp mask).

    Усиливает контраст на границах, вычитая размытую версию изображения из
    исходной и добавляя разницу обратно с заданной силой, что искажает
    локальную структуру и может подавлять водяной знак.
    """

    @staticmethod
    def attack(args: dict = {
                "input_image": [[[0]]],
                "amount": 1.0,
                "radius": 2.0,
                "threshold": 0.0
    }):
        """
        Применяет нерезкое маскирование к изображению и возвращает результат.

        Args:
            args (dict): параметры атаки
                input_image (list(list(list(int)))): матрица изображения
                amount (float): сила эффекта резкости, диапазон [0.1, 5.0] (по умолчанию 1.0)
                radius (float): радиус размытия маски, диапазон [0.5, 5.0] (по умолчанию 2.0)
                threshold (float): порог срабатывания в долях канала 0..1, диапазон [0.0, 1.0] (по умолчанию 0.0)

        Returns:
            output_image (list(list(list(int)))): матрица изображения после атаки

        Raises:
            ValueError: если amount вне диапазона [0.1, 5.0], radius вне диапазона [0.5, 5.0]
                или threshold вне диапазона [0.0, 1.0]
        """
        input_image = args["input_image"]
        amount = float(args.get("amount", 1.0))
        radius = float(args.get("radius", 2.0))
        threshold = float(args.get("threshold", 0.0))

        if not (0.1 <= amount <= 5.0):
            raise ValueError("amount must be in range [0.1-5.0]")
        if not (0.5 <= radius <= 5.0):
            raise ValueError("radius must be in range [0.5-5.0]")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be in range [0.0-1.0]")

        img = Image.fromarray(np.array(input_image, dtype=np.uint8), "RGB")
        sharp_img = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(amount * 100), threshold=int(threshold * 255)))
        return np.array(sharp_img).tolist()