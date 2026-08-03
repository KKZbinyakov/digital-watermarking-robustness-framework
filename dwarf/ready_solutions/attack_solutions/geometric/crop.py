"""Атака кадрирования: вырезает часть кадра и восстанавливает исходный размер."""

from ...utils.attack_utils import *

_ANCHORS = {
    "center": ("middle", "middle"),
    "top_left": ("start", "start"),
    "top_right": ("start", "end"),
    "bottom_left": ("end", "start"),
    "bottom_right": ("end", "end"),
    "random": ("random", "random"),
}
"""Положение сохраняемой области: якоря по вертикали и по горизонтали."""

_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}
"""Фильтры интерполяции Pillow, допустимые в режиме 'resize'."""

_MODES = ("pad", "resize", "raw")
"""Способы восстановления исходного размера кадра."""


def _side(extent: int, ratio: float) -> int:
    """
    Считает длину сохраняемой стороны.

    Результат прижимается снизу к одному пикселю: на кадрах в несколько пикселей
    округление вниз даёт срез нулевой ширины, из-за которого атака падает вместо
    того, чтобы выродиться в тождественную.

    Args:
        extent (int): длина стороны кадра в пикселях
        ratio (float): доля сохраняемой стороны, 0 < ratio <= 1

    Returns:
        int: длина сохраняемой стороны, 1..extent
    """
    return max(1, min(extent, int(round(extent * ratio))))


def _origin(anchor: str, extent: int, kept: int, rng) -> int:
    """
    Считает координату начала сохраняемой области по якорю.

    Args:
        anchor (str): 'start', 'middle', 'end' или 'random'
        extent (int): длина стороны кадра в пикселях
        kept (int): длина сохраняемой стороны в пикселях
        rng (np.random.Generator): генератор случайных чисел для якоря 'random'

    Returns:
        int: координата начала области, 0..extent - kept
    """
    if anchor == "start":
        return 0
    if anchor == "end":
        return extent - kept
    if anchor == "middle":
        return (extent - kept) // 2
    return int(rng.integers(0, extent - kept + 1))


def _fill_color(fill) -> np.ndarray:
    """
    Приводит цвет заливки к тройке уровней RGB.

    Args:
        fill (int | float | Sequence): уровень серого либо тройка RGB со значениями 0..255

    Returns:
        np.ndarray: массив (3,) типа uint8

    Raises:
        ValueError: если fill не уровень и не тройка либо уровни вне диапазона 0..255
    """
    if isinstance(fill, (int, float, np.integer, np.floating)):
        levels = (fill, fill, fill)
    else:
        try:
            levels = tuple(fill)
        except TypeError as error:
            raise ValueError(
                f"fill должен быть уровнем 0..255 или тройкой RGB, получено {fill!r}"
            ) from error

    if len(levels) != 3:
        raise ValueError(
            f"fill должен быть уровнем 0..255 или тройкой RGB, получено {fill!r}"
        )
    if any(not 0 <= level <= 255 for level in levels):
        raise ValueError(
            f"уровни fill должны быть в диапазоне 0..255, получено {fill!r}"
        )

    return np.round(levels).astype(np.uint8)


class Crop(Ready_Geometric_Attacks):
    """
    Атака кадрирования.

    Сохраняет прямоугольную область кадра и отбрасывает всё за её границами.
    Исходный размер при этом восстанавливается: либо заливкой освободившегося
    места, либо растяжением сохранённой области.
    синхронизации.
    """

    @staticmethod
    def attack(
        args: dict = {
            "input_image": [[[0]]],
            "ratio": 0.5,
            "position": "center",
            "mode": "pad",
            "fill": 0,
            "resample": "bicubic",
            "seed": None,
        }
    ):
        """
        Кадрирует изображение и сохраняет результат.

        Args:
            args (dict): параметры атаки
                input_image (list(list(list(int))): матрица изображения
                ratio (float): доля сохраняемой стороны по каждой оси, 0 < ratio <= 1;
                    при 0.5 остаётся четверть площади (по умолчанию 0.5)
                position (str): положение области: 'center', 'top_left', 'top_right',
                    'bottom_left', 'bottom_right' или 'random' (по умолчанию 'center')
                mode (str): восстановление размера: 'pad' — заливка отброшенного,
                    'resize' — растяжение области до исходного размера, 'raw' —
                    вернуть обрезанный кадр как есть (по умолчанию 'pad')
                fill (int | tuple): уровень или тройка RGB для заливки в режиме 'pad'
                    (по умолчанию 0)
                resample (str): фильтр интерполяции для режима 'resize': 'nearest',
                    'bilinear', 'bicubic' или 'lanczos' (по умолчанию 'bicubic')
                seed (int): зерно генератора случайных чисел для position='random'
                    (по умолчанию None)

        Returns:
            output_image (list(list(list(int)))): матрица изображения после атаки

        Raises:
            ValueError: если ratio вне диапазона (0, 1] либо position, mode, resample
                или fill недопустимы
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        ratio = float(args.get("ratio", 0.5))
        position = args.get("position", "center")
        mode = args.get("mode", "pad")
        resample = args.get("resample", "bicubic")
        fill = args.get("fill", 0)
        seed = args.get("seed", None)

        if not 0 < ratio <= 1:
            raise ValueError(f"ratio должен быть в диапазоне (0, 1], получено {ratio}")
        if position not in _ANCHORS:
            raise ValueError(
                f"Неизвестный position={position!r}, ожидается один из "
                f"{', '.join(sorted(_ANCHORS))}"
            )
        if mode not in _MODES:
            raise ValueError(
                f"Неизвестный mode={mode!r}, ожидается один из {', '.join(_MODES)}"
            )
        if mode == "resize" and resample not in _RESAMPLING:
            raise ValueError(
                f"Неизвестный resample={resample!r}, ожидается один из "
                f"{', '.join(sorted(_RESAMPLING))}"
            )
        color = _fill_color(fill)

        array = load_rgb(input_data)
        height, width = array.shape[:2]
        kept_height = _side(height, ratio)
        kept_width = _side(width, ratio)

        rng = np.random.default_rng(seed)
        anchor_y, anchor_x = _ANCHORS[position]
        top = _origin(anchor_y, height, kept_height, rng)
        left = _origin(anchor_x, width, kept_width, rng)

        region = np.ascontiguousarray(
            array[top : top + kept_height, left : left + kept_width]
        )

        if mode == "raw":
            return region
        """
        if mode == "resize":
            restored = Image.fromarray(region, "RGB").resize(
                (width, height), _RESAMPLING[resample]
            )
            return restored
        """
        canvas = np.empty_like(array)
        canvas[:, :] = color
        canvas[top : top + kept_height, left : left + kept_width] = region
        return canvas
