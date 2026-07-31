from ...utils.attack_utils import *

class Unsharp_Mask(Ready_Filtering_Attacks):
    """
    Атака нерезким маскированием (unsharp mask).

    Усиливает контраст на границах, вычитая размытую версию изображения из
    исходной и добавляя разницу обратно с заданной силой, что искажает
    локальную структуру и может подавлять водяной знак.
    """

    @staticmethod
    def attack(args: dict = {
                "input_data": None,
                "output_data": None
    }):
        """
        Применяет нерезкое маскирование к изображению и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                amount (float): сила эффекта резкости, диапазон [0.1, 5.0] (по умолчанию 1.0)
                radius (float): радиус размытия маски, диапазон [0.5, 5.0] (по умолчанию 2.0)
                threshold (float): порог срабатывания в долях канала 0..1, диапазон [0.0, 1.0] (по умолчанию 0.0)

        Returns:
            None

        Raises:
            ValueError: если amount вне диапазона [0.1, 5.0], radius вне диапазона [0.5, 5.0]
                или threshold вне диапазона [0.0, 1.0]
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        amount = float(args.get("amount", 1.0))
        radius = float(args.get("radius", 2.0))
        threshold = float(args.get("threshold", 0.0))
        
        if not (0.1 <= amount <= 5.0):
            raise ValueError("amount должен быть в диапазоне [0.1-5.0]")
        if not (0.5 <= radius <= 5.0):
            raise ValueError("radius должен быть в диапазоне [0.5-5.0]")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold должен быть в диапазоне [0.0-1.0]")
        
        img = Image.open(input_data).convert("RGB")
        sharp_img = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(amount * 100), threshold=int(threshold * 255)))
        sharp_img.save(output_data)