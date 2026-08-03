from dwarf.ready_solutions.utils.attack_utils import *
from scipy.signal import wiener

class Wiener_Filter(Ready_Filtering_Attacks):
    """
    Атака винеровской фильтрацией.

    Применяет к каждому каналу изображения адаптивный винеровский фильтр,
    сглаживающий изображение с учётом локальной дисперсии и оценки шума,
    что может подавлять слабый сигнал водяного знака.
    """

    @staticmethod
    def attack(args: dict = {
                    "input_image": [[[0]]],
                    "window": 5,
                    "noise": 0.0
    }):
        """
        Применяет винеровский фильтр к каждому каналу изображения и возвращает результат.

        Args:
            args (dict): параметры атаки
                input_image (list(list(list(int)))): матрица изображения
                window (int): размер окна фильтра, нечётное число >= 3 (по умолчанию 5)
                noise (float): оценка дисперсии шума, при 0.0 оценивается автоматически по локальной дисперсии (по умолчанию 0.0)

        Returns:
            output_image (list(list(list(int)))): матрица изображения после атаки

        Raises:
            ValueError: если window меньше 3 или чётное, либо noise отрицательный
        """
        input_image = args["input_image"]
        window = args.get("window", 5)
        noise = float(args.get("noise", 0.0))

        if window < 3 or window % 2 == 0:
            raise ValueError("Window size must be an odd number >= 3")
        if noise is not None and noise < 0:
            raise ValueError("Noise estimate cannot be negative")

        data = np.array(input_image, dtype=np.float32) / 255.0

        filtered = np.empty_like(data)
        for c in range(data.shape[2]):
            filtered[..., c] = wiener(data[..., c], mysize=(window, window), noise=noise)

        filtered = np.nan_to_num(filtered, nan=0.0)
        filtered = np.clip(filtered, 0, 1)
        return (filtered * 255).astype(np.uint8).tolist()