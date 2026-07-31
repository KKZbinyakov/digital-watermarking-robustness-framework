from ...utils.attack_utils import *

class Anisotropic_Diffusion(Ready_Filtering_Attacks):
    """
    Атака анизотропной диффузией (фильтр Перона-Малик).

    Итеративно сглаживает изображение нелинейной диффузией: внутри однородных
    областей диффузия идёт почти как изотропное размытие и стирает мелкие
    детали и шум, а на резких границах коэффициент проводимости падает почти
    до нуля и граница сохраняется. За счёт этого фильтр подавляет шум и
    мелкоструктурный водяной знак, искажая крупные контуры заметно меньше,
    чем гауссово или прямоугольное размытие той же силы.
    """

    @staticmethod
    def attack(args: dict = {
            "input_data": None,
            "output_data": None
    }):
        """
        Применяет анизотропную диффузию Перона-Малик к изображению и сохраняет результат.

        Диффузия ведётся одновременно по всем трём каналам RGB на кадре,
        нормированном к диапазону 0..1. Края кадра обрабатываются отражением
        (продолжением крайних отсчётов), без циклического оборачивания.

        Args:
            args (dict): параметры атаки
                input_data (str): путь к исходному изображению
                output_data (str): путь для сохранения результата
                iterations (int): число итераций диффузии, диапазон [1, 50] (по умолчанию 15)
                kappa (float): порог проводимости в долях канала 0..1, диапазон [0.01, 0.5] (по умолчанию 0.1)
                gamma (float): шаг диффузии за итерацию, диапазон (0.0, 0.25] (по умолчанию 0.2)
                option (int): функция проводимости: 1 — экспоненциальная, сильнее сохраняет резкие
                    границы; 2 — рациональная, слабее подавляет широкие плавные перепады (по умолчанию 1)

        Returns:
            None

        Raises:
            ValueError: если iterations вне диапазона [1, 50], kappa вне диапазона [0.01, 0.5],
                gamma вне диапазона (0.0, 0.25] или option не равен 1 или 2
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        iterations = int(args.get("iterations", 15))
        kappa = float(args.get("kappa", 0.1))
        gamma = float(args.get("gamma", 0.2))
        option = int(args.get("option", 1))

        if not (1 <= iterations <= 50):
            raise ValueError("iterations должен быть в диапазоне [1-50]")
        if not (0.01 <= kappa <= 0.5):
            raise ValueError("kappa должна быть в диапазоне [0.01-0.5]")
        if not (0.0 < gamma <= 0.25):
            raise ValueError("gamma должна быть в диапазоне (0.0-0.25]")
        if option not in (1, 2):
            raise ValueError("option должен быть 1 или 2")

        img = Image.open(input_data).convert("RGB")
        data = np.array(img, dtype=np.float32) / 255.0

        for _ in range(iterations):
            padded = np.pad(data, ((1, 1), (1, 1), (0, 0)), mode="edge")

            north = padded[:-2, 1:-1, :] - data
            south = padded[2:, 1:-1, :] - data
            east = padded[1:-1, 2:, :] - data
            west = padded[1:-1, :-2, :] - data

            if option == 1:
                c_n = np.exp(-(north / kappa) ** 2)
                c_s = np.exp(-(south / kappa) ** 2)
                c_e = np.exp(-(east / kappa) ** 2)
                c_w = np.exp(-(west / kappa) ** 2)
            else:
                c_n = 1.0 / (1.0 + (north / kappa) ** 2)
                c_s = 1.0 / (1.0 + (south / kappa) ** 2)
                c_e = 1.0 / (1.0 + (east / kappa) ** 2)
                c_w = 1.0 / (1.0 + (west / kappa) ** 2)

            data = data + gamma * (c_n * north + c_s * south + c_e * east + c_w * west)

        data = np.clip(data, 0, 1)
        filtered_img = Image.fromarray((data * 255).astype(np.uint8))
        filtered_img.save(output_data)
