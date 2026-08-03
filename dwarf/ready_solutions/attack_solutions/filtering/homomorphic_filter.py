from dwarf.ready_solutions.utils.attack_utils import *

class Homomorphic_Filter(Ready_Filtering_Attacks):
    """
    Атака гомоморфной фильтрацией.

    Разделяет яркостный канал на низкочастотную (освещённость) и
    высокочастотную (отражение) составляющие через логарифм и Фурье-образ,
    затем по-разному их усиливает, что переупорядочивает частотный состав
    изображения и подавляет спрятанный в нём водяной знак.
    """

    @staticmethod
    def attack(args: dict = {
                        "input_image": [[[0]]],
                        "gamma_low": 0.5,
                        "gamma_high": 2.0,
                        "cutoff": 32.0,
                        "c": 1.0
    }):
        """
        Применяет гомоморфную фильтрацию к яркостному каналу изображения и возвращает результат.

        Работает в пространстве YCbCr: фильтруется только канал Y, каналы
        цветности не изменяются.

        Args:
            args (dict): параметры атаки
                input_image (list(list(list(int)))): матрица изображения
                gamma_low (float): коэффициент усиления низких частот, диапазон [0.1, 1.0] (по умолчанию 0.5)
                gamma_high (float): коэффициент усиления высоких частот, диапазон [1.0, 3.0] (по умолчанию 2.0)
                cutoff (float): частота среза фильтра, диапазон [10.0, 100.0] (по умолчанию 32.0)
                c (float): коэффициент крутизны перехода фильтра (по умолчанию 1.0)

        Returns:
            output_image (list(list(list(int)))): матрица изображения после атаки

        Raises:
            ValueError: если gamma_low вне диапазона [0.1, 1.0], gamma_high вне диапазона [1.0, 3.0]
                или cutoff вне диапазона [10.0, 100.0]
        """
        input_image = args["input_image"]
        gamma_low = float(args.get("gamma_low", 0.5))
        gamma_high = float(args.get("gamma_high", 2.0))
        cutoff = float(args.get("cutoff", 32.0))
        c = float(args.get("c", 1.0))

        if not (0.1 <= gamma_low <= 1.0):
            raise ValueError("gamma_low must be in range [0.1-1.0]")
        if not (1.0 <= gamma_high <= 3.0):
            raise ValueError("gamma_high must be in range [1.0-3.0]")
        if not (10.0 <= cutoff <= 100.0):
            raise ValueError("cutoff must be in range [10-100]")

        img = Image.fromarray(np.array(input_image, dtype=np.uint8), "RGB")

        ycbcr = np.array(img.convert("YCbCr"), dtype=np.float32)
        Y = ycbcr[..., 0] / 255.0
        height, width = Y.shape
        
        log_Y = np.log1p(Y)

        F = np.fft.fftshift(np.fft.fft2(log_Y))

        u = np.arange(height) - height / 2
        v = np.arange(width) - width / 2
        V, U = np.meshgrid(v, u)
        D2 = U**2 + V**2
        H = (gamma_high - gamma_low) * (1.0 - np.exp(-c * D2 / (cutoff**2))) + gamma_low
        
        filtered = np.fft.ifft2(np.fft.ifftshift(F * H))
        filtered = np.real(filtered)

        result_Y = np.expm1(filtered)

        result_Y = (result_Y - result_Y.min()) / (result_Y.max() - result_Y.min() + 1e-8)

        ycbcr[..., 0] = np.clip(result_Y * 255.0, 0, 255)
        out_img = Image.fromarray(ycbcr.astype(np.uint8), mode="YCbCr").convert("RGB")

        return np.array(out_img).tolist()