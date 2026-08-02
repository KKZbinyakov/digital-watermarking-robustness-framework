from ...utils.attack_utils import *


class Bit_Depth_Reduction(Ready_Color_Brightness_Attacks):
    """
    Атака уменьшения битовой глубины.

    Оставляет только старшие биты каждого канала и растягивает полученные уровни
    обратно на диапазон 0..255
    """

    @staticmethod
    def attack(args: dict = {"input_data": None, "output_data": None}):
        """
        Огрубляет битовую глубину изображения и сохраняет результат.

        Результат зависит только от исходного уровня пикселя, поэтому
        преобразование считается один раз на таблицу из 256 значений.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                bits (int): сколько старших бит оставить, 1..8 (по умолчанию 4)

        Returns:
            None

        Raises:
            ValueError: если bits вне диапазона 1..8
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        bits = int(args.get("bits", 4))

        if not 1 <= bits <= 8:
            raise ValueError(f"bits должен быть в диапазоне 1..8, получено {bits}")

        shift = 8 - bits
        levels = (1 << bits) - 1
        lookup = (np.arange(256, dtype=np.uint16) >> shift) / levels * 255.0
        lookup = np.clip(lookup, 0, 255).round().astype(np.uint8)
        save_rgb(lookup[load_rgb(input_data)], output_data)
