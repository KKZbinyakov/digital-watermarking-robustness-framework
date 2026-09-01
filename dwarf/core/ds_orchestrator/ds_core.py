import json
import multiprocessing
import os
import time
from pathlib import Path
from queue import Full

import numpy as np
from PIL import Image

from dwarf.core.dwarf_exceptions import Dwarf_Exception
from dwarf.core.utils.utils import file_to_hash


class Ds_Core:
    """
    Общая логика для датасета. На пайплайне - загрузчик изображений. Анализирует директорию датасета, загружает картинки, приводит их к нужной конфигурации, загружает в очередь пак данных. Все проверки, которые есть на данный момент - условные и могут не нести смысла.

    Args:
        config (dict): Конфигурация

    Attributes:
        queue (multiprocessing.Queue): Очередь
        queues_state (list): Состояние очередей
        source (str): Путь к директории с датасетом
        image_type (type): Тип изображения
        image_color_model (str): Модель цвета
        image_geometry (list): Геометрия изображения
        extensions (list): Список расширений

        MANIFEST_FILE_NAME (str): Имя/путь файла манифеста
        FAILOVER_FILE (str): Имя/путь файла failover.json
        RETRY_COUNT (int): Количество попыток отправить данные в очередь
        BUFFER_LIMIT (int): Лимит локального буфера

        local_buffers (list): Локальные буферы


    Methods:
        check_state(): Проверка соответствия конфигурации нужному состоянию. Необходимо для реализации, поскольку слишком много атрибутов нужно учесть.
        generate_file_manifest(): Генерация манифеста изображений, это будет что-то типо .txt файла, где на каждой новой строке будет путь к изображению (+ другие данные). Вроде как индустриальный стандарт, удобно будет работать, нужно будет создать его один раз в самом начале.
        image_to_type(): Приведение изображения к нужному типу
        image_to_color_model(): Приведение изображения к нужной модели цвета
        image_to_correct_geometry(): Приведение изображения к нужной геометрии
        formatter(): Формирование пака данных, соответственно здесь мы переводим картинку из пути в Image, приводим к нужному типу, модели цвета и геометрии, а потом превращаем в numpy array
        save_buffer_to_file(): Сохранение локального буфера в файл
        try_send_with_buffer(): Попытка отправить данные в очередь
        process_failover(): Повторная отправка данных из failover.json
        send_dead_pill(): Отправка dead pill в очереди
        main_loop(): Основной цикл

    Returns:
        None

    Raises:
        Dwarf_Exception: Ошибка
    """
    def __init__(self, config: dict):
        self.queues: list[multiprocessing.Queue] = config.get("queue")
        self.queues_state: list[bool] = None #= [True for _ in range(len(self.queues))]
        self.source: str = config.get("source")
        self.image_type: type = config.get("image_type")
        self.image_color_model: str = config.get("image_color_model")
        self.image_geometry: list = config.get("image_geometry")
        self.extensions: list = [".png", ".jpg", ".bmp"]
        self.MANIFEST_FILE_NAME = "manifest.json"
        self.FAILOVER_FILE = "failover.json"
        self.RETRY_COUNT = 3
        self.BUFFER_LIMIT = 10
        self.local_buffers = None #[[] for _ in range(len(self.queues))]

    def check_state(self) -> bool:
        """
        Проверка соответствия конфигурации нужному состоянию

        Returns:
            bool: True если всё хорошо, False в противном случае
        """
        try:
            path = Path(self.source)
            if not (path.exists() and path.is_dir()):
                raise Dwarf_Exception(f"Path {path} does not exist or is not a directory")
            for queue in self.queues:
                if not isinstance(queue, multiprocessing.Queue):
                    raise Dwarf_Exception(f"Queue {queue} is not a multiprocessing.Queue")
            self.queues_state = [True for _ in range(len(self.queues))]
            self.local_buffers = [[] for _ in range(len(self.queues))]
            self.MANIFEST_FILE_NAME = self.source + "/" + self.MANIFEST_FILE_NAME
            self.FAILOVER_FILE = self.source + "/" + self.FAILOVER_FILE

        except Exception as e:
            raise Dwarf_Exception(f"Failed to check state: {e}") from e
        return True

    def generate_file_manifest(self) -> bool:
        folder = Path(self.source)
        try:
            with open(self.MANIFEST_FILE_NAME, "w") as f:
                for extension in self.extensions:
                    for file_path in folder.rglob("*" + extension):
                        abs_path = file_path.absolute()
                        file_info = {
                            "name": file_path.name,
                            "path": str(abs_path),
                            "hash": file_to_hash(abs_path)
                        }
                        f.write(json.dumps(file_info) + "\n")
            return True
        except Exception as e:
            raise Dwarf_Exception(f"Failed to generate manifest: {e}") from e

    def image_to_type(self, image: Image) -> np.ndarray:
        """
        Приведение изображения к нужному типу

        Args:
            image (Image): Изображение

        Returns:
            np.ndarray: Изображение в виде numpy array

        Raises:
            Dwarf_Exception: Ошибка
        """
        try:
            match self.image_type:
                case np.float32:
                    img_numpy = np.array(image, dtype=np.float32)
                case np.float64:
                    img_numpy = np.array(image, dtype=np.float64)
                case np.int32:
                    img_numpy = np.array(image, dtype=np.int32)
                case np.int64:
                    img_numpy = np.array(image, dtype=np.int64)
                case np.uint32:
                    img_numpy = np.array(image, dtype=np.uint32)
                case np.uint64:
                    img_numpy = np.array(image, dtype=np.uint64)
                case _:
                    raise Dwarf_Exception("Incorrect image type")
            return img_numpy
        except Exception as e:
            raise Dwarf_Exception(f"Failed to convert image to type: {e}") from e

    def image_to_color_model(self, image: Image) -> Image:
        """
        Приведение изображения к нужному цветовому режиму

        Args:
            image (Image): Изображение

        Returns:
            Image: Изображение в нужном цветовом режиме

        Raises:
            Dwarf_Exception: Ошибка
        """
        try:
            match self.image_color_model:
                case "RGB":
                    image = image.convert("RGB")
                case "RGBA":
                    image = image.convert("RGBA")
                case "L":
                    image = image.convert("L")
                case "CMYK":
                    image = image.convert("CMYK")
                case _:
                    raise Dwarf_Exception("Incorrect color model")
            return image
        except Exception as e:
            raise Dwarf_Exception(f"Failed to convert image to color model: {e}") from e

    def image_to_correct_geometry(self, image: Image) -> Image:
        """
        Приведение изображения к нужному размеру

        Args:
            image (Image): Изображение

        Returns:
            Image: Изображение в нужном размере

        Raises:
            Dwarf_Exception: Ошибка
        """
        try:
            img = image.resize(self.image_geometry)
            return img
        except Exception as e:
            raise Dwarf_Exception(f"Failed to convert image to correct geometry: {e}") from e

    def formatter(self, image: Image) -> np.ndarray:
        """
        Форматирование изображения

        Args:
            image (Image): Изображение

        Returns:
            np.ndarray: Форматированное изображение

        Raises:
            Dwarf_Exception: Ошибка
        """
        try:
            image = self.image_to_color_model(image)
            image = self.image_to_correct_geometry(image)
            image = self.image_to_type(image)
            return image
        except Exception as e:
            raise Dwarf_Exception(f"Failed to format image: {e}") from e

    def save_buffer_to_file(self, target_idx):
        """
        Сохранение буфера в файл

        Args:
            target_idx (int): Индекс целевой очереди

        Raises:
            Dwarf_Exception: Ошибка
        """
        try:
            with open(self.FAILOVER_FILE + "_" + str(target_idx), "a") as f:
                for image in self.local_buffers[target_idx]:
                    f.write(json.dumps(image) + "\n")
        except Exception as e:
            raise Dwarf_Exception(f"Failed to save buffer to file: {e}") from e

    def try_send_with_buffer(self, target_idx, data: dict):
        """
        Попытка отправить данные в очередь

        Args:
            target_idx (int): Индекс целевой очереди
            data (dict): Данные для отправки

        Returns:
            bool: True, если данные успешно отправлены, False в противном случае
        """
        try:
            if not self.queues_state[target_idx]:
                raise Dwarf_Exception(f"Queue {target_idx} is already dead")
            queue = self.queues[target_idx]
            try:
                queue.put(data, timeout=0.01)
                return True
            except Full:
                self.local_buffers[target_idx].append(data)
                if len(self.local_buffers[target_idx]) >= self.BUFFER_LIMIT:
                    self.save_buffer_to_file(target_idx)
                    self.local_buffers[target_idx].clear()
            except (BrokenPipeError, OSError):
                self.queues_state[target_idx] = False
                # Добавить лог ошибки
        except Exception as e:
            raise Dwarf_Exception(f"Failed to send data to queue {queue}: {e}") from e

    def process_failover(self):
        """
        Обработка failover

        Raises:
            Dwarf_Exception: Ошибка
        """
        for target_idx in range(len(self.queues)):
            try:
                with open(self.FAILOVER_FILE + "_" + str(target_idx)) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            try:
                                data = json.loads(line)
                            except Exception:
                                # raise Dwarf_Exception(f"Failed to load failover of file {line}: {e}")
                                continue
                            queue = self.queues[target_idx]
                            if not self.queues_state[target_idx]:
                                raise Dwarf_Exception(f"Queue {target_idx} is already dead, failover is not running")
                            for _i in range(self.RETRY_COUNT):
                                try:
                                    queue.put(data, timeout=0.01)
                                    break
                                except Full:
                                    time.sleep(0.1)
                                    continue
                                except (BrokenPipeError, OSError):
                                    self.queues_state[target_idx] = False
                                    break
                        except Exception as e:
                            raise Dwarf_Exception(f"Failed to load manifest of file {line} after {self.RETRY_COUNT} tries: {e}") from e
                os.remove(self.FAILOVER_FILE + "_" + str(target_idx))
            except Exception as e:
                raise Dwarf_Exception(f"Failed to process failover for queue {target_idx}: {e}") from e

    def send_dead_pill(self):
        """
        Отправка сигнала о завершении в очередь

        Raises:
            Dwarf_Exception: Ошибка
        """
        try:
            for target_idx in range(len(self.queues)):
                if not self.queues_state[target_idx]:
                    continue
                path = self.FAILOVER_FILE + "_" + str(target_idx)
                if os.path.exists(path):
                    print(f"File {path} still exists, skipping")
                    continue
                else:
                    try:
                        self.queues[target_idx].put(None, timeout=0.5)
                        self.queues_state[target_idx] = False
                    except Exception as e:
                        raise Dwarf_Exception(f"Failed to send dead pill to queue {target_idx}: {e}") from e
        except Exception as e:
            raise Dwarf_Exception(f"Failed to send dead pill: {e}") from e

    def main_loop(self):
        """
        Основной цикл

        Raises:
            Dwarf_Exception: Ошибка
        """
        if not self.check_state():
            raise Dwarf_Exception("State is incorrect, main loop is not running")

        if not self.generate_file_manifest():
            raise Dwarf_Exception("Failed to generate manifest")

        try:
            manifest_file_path = self.MANIFEST_FILE_NAME
            with open(manifest_file_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data["hash"] != file_to_hash(data["path"]):
                            print(f"Hash of file {data['path']} is incorrect")
                            continue
                        raw_image = Image.open(data["path"])
                        numpy_image = self.formatter(raw_image)
                            #raise Dwarf_Exception(f"Hash of file {data['path']} is incorrect")

                        data_to_send = {
                            "name": data["name"],
                            "image": numpy_image
                        }

                        for target_idx in range(len(self.queues)):
                            self.try_send_with_buffer(target_idx, data_to_send)

                    except Exception as e:
                        print(f"Failed to load manifest of file {line}: {e}")
                        continue
                        # raise Dwarf_Exception(f"Failed to load manifest of file {line}: {e}")
            for idx in range(len(self.local_buffers)):
                if self.local_buffers[idx]:
                    self.save_buffer_to_file(idx)
                    self.local_buffers[idx].clear()

            self.process_failover()
            self.send_dead_pill()
        except Exception as e:
            raise Dwarf_Exception(f"Failed to run main loop: {e}") from e
        finally:
            try:
                for target_idx in range(len(self.queues)):
                    self.save_buffer_to_file(target_idx)
            except Exception as e:
                self.send_dead_pill()
                raise Dwarf_Exception(f"Failed to save buffer to file in finally: {e}") from e
            try:
                self.process_failover()
            except Exception as e:
                self.send_dead_pill()
                raise Dwarf_Exception(f"Failed to process failover in finally: {e}") from e
            try:
                self.send_dead_pill()
            except Exception as e:
                raise Dwarf_Exception(f"Failed to send dead pill in finally: {e}") from e
