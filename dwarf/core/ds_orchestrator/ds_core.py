import multiprocessing

from PIL import Image

from dwarf.core.dwarf_exceptions import Dwarf_Exception


class Ds_Core:
    """
    Общая логика для датасета. На пайплайне - загрузчик изображений. Анализирует директорию датасета, загружает картинки, 
    приводит их к нужной конфигурации, загружает в очередь пак данных. Все проверки, которые есть на данный момент - условные и могут не нести смысла.

    Args:
        config (dict): Конфигурация

    Attributes:
        queue (multiprocessing.Queue): Очередь
        source (str): Путь к директории с датасетом
        images (dict): Словарь изображений
        data (dict): Пак данных
        image_type (type): Тип изображения
        image_color_model (str): Модель цвета
        image_geometry (list): Геометрия изображения
        extensions (list): Список расширений, пока хз, точно ли он нам нужен, можем типо отсеевать неподходящие

    Methods:
        check_state(): Проверка соответствия конфигурации нужному состоянию. Необходимо для реализации, поскольку слишком много атрибутов нужно учесть.
        generate_file_manifest(): Генерация манифеста изображений, это будет что-то типо .txt файла, где на каждой новой строке будет путь к изображению (+ другие данные). 
        Вроде как индустриальный стандарт, удобно будет работать, нужно будет создать его один раз в самом начале.
        load_in_queue(): Загрузка изображений в очередь
        image_to_type(): Приведение изображения к нужному типу
        image_to_color_model(): Приведение изображения к нужной модели цвета
        image_to_correct_geometry(): Приведение изображения к нужной геометрии
        formatter(): Формирование пака данных, соответственно здесь мы переводим картинку из пути в Image, приводим к нужному типу, модели цвета и геометрии, а потом превращаем в numpy array
        main_loop(): Главный цикл, здесь вся логика.

    Returns:
        None

    Raises:
        Dwarf_Exception: Ошибка
    """
    def __init__(self, config: dict):
        self.queue: multiprocessing.JoinableQueue = config.get("queue")
        self.source: str = config.get("source")
        self.images: dict = None
        self.data: dict = None
        self.image_type: type = config.get("image_type")
        self.image_color_model: str = config.get("image_color_model")
        self.image_geometry: list = config.get("image_geometry")
        self.extensions: list = [".png", ".jpg", ".bmp"]
        pass

    def check_state(self):
        return

    def generate_file_manifest(self):
        if self.source is None:
            raise Dwarf_Exception("No source of directory")
        return

    def load_in_queue(self, data: dict):
        if type(self.queue) is not multiprocessing.Queue:
            raise Dwarf_Exception("Queue is incorrect type")
        return

    def image_to_type(self, image: Image):
        if self.image_type is None or type(self.image_type) is not type:
            raise Dwarf_Exception("Incorrect image type")
        return image

    def image_to_color_model(self, image: Image):
        return image

    def image_to_correct_geometry(self, image: Image):
        try:
            if type(self.image_geometry[0]) is not int and type(self.image_geometry[0]) is not int:
                raise Dwarf_Exception("Incorrect geometry value type")
            return image
        except Exception as e:
            raise e

    def formatter(self):
            if self.images is None:
                raise Dwarf_Exception("No images are loaded")
            return

    def main_loop(self):
        pass
