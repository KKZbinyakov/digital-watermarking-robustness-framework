from src.dwarf.ready_solutions.utils.utils import *

class Crop(Ready_Geometric_Attacks):
    @staticmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }):
        """
        Оставляет правую верхнюю четверть изображения (ширина/2, высота/2).
        :param image_path: путь к исходному изображению
        :param output_path: путь для сохранения обрезанного изображения
        """
        input_data = args["input_data"]
        output_data = args["output_data"]
        img = Image.open(input_data)
        width, height = img.size
        # правый верхний угол: x от width//2 до width, y от 0 до height//2
        left = width // 2
        top = 0
        right = width
        bottom = height // 2
        cropped = img.crop((left, top, right, bottom))
        cropped.save(output_data)
        print(f"Атака crop выполнена: сохранено {cropped.size} в {output_data}")
    def crop_attack(image_path, output_path):
        """
        Оставляет правую верхнюю четверть изображения (ширина/2, высота/2).
        :param image_path: путь к исходному изображению
        :param output_path: путь для сохранения обрезанного изображения
        """
        img = Image.open(image_path)
        width, height = img.size
        # правый верхний угол: x от width//2 до width, y от 0 до height//2
        left = width // 2
        top = 0
        right = width
        bottom = height // 2
        cropped = img.crop((left, top, right, bottom))
        cropped.save(output_path)
        print(f"Атака crop выполнена: сохранено {cropped.size} в {output_path}")