import json
import multiprocessing
import sqlite3
import time
from multiprocessing import shared_memory
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from pathlib import Path
from queue import Empty, Full

import numpy as np
from PIL import Image, ImageOps

from dwarf.core.dwarf_exceptions import Dwarf_Exception
from dwarf.core.logs.logger import logger


def make_shm_name(run_id: str, image_id: str) -> str:
    return f"dwarf_{run_id}_{image_id}"[:255]

DEAD_PILL = None

class Ds_Core:
    """
    Общая логика для датасета. На пайплайне - загрузчик изображений. Анализирует директорию датасета, загружает картинки, приводит их к нужной конфигурации, загружает в очередь пак данных. Все проверки, которые есть на данный момент - условные и могут не нести смысла.

    Args:
        config (dict): Конфигурация

    Attributes:
        queue (multiprocessing.Queue): Очередь
        queue_state: Состояние очереди
        source (str): Путь к директории с датасетом
        image_type (type): Тип изображения
        image_color_model (str): Модель цвета
        image_geometry (list): Геометрия изображения
        extensions (list): Список расширений

        MANIFEST_FILE_NAME (str): Имя файла манифеста
        MANIFEST_FILE_PATH (str): Путь к файлу манифеста
        FAILOVER_FILE_PATH (str): Путь к файлу failover.json
        RETRY_COUNT (int): Количество попыток отправить данные в очередь
        BUFFER_LIMIT (int): Лимит локального буфера

        local_buffer: Локальный буфер


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
        self.queue_ready: multiprocessing.Queue = config.get("queue_ready")
        self.queue_free: multiprocessing.Queue = config.get("queue_free")
        self.queue_state: bool = True
        self.source: str = config.get("source")
        self.image_type: type = np.uint8
        self.image_color_model: str = config.get("image_color_model")
        self.image_geometry: list = config.get("image_geometry")
        self.image_color_models: list[str] = ["RGB", "RGBA", "CMYK", "L"]
        self.extensions: list = [".png", ".jpg", ".bmp"]
        self.MANIFEST_FILE_PATH = config.get("manifest_file_path")
        self.RETRY_COUNT = 3
        self.BUFFER_LIMIT = 10
        self.local_buffer = []
        self.stop_event = config.get("stop_event")
        self.run_id = config.get("run_id")
        self.image_id = None
        self.shm_refs: dict[str, shared_memory.SharedMemory] = {}
        self.exif_transpose = config.get("exif_transpose", True)
        self.resample = config.get("resample", "lanczos")
        self.db_conn = None
        self.database_path = config.get("database_path")

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
            if not isinstance(self.queue_ready, MPQueue):
                raise Dwarf_Exception("config['queue'] is not a multiprocessing.Queue")
            if self.image_color_model not in self.image_color_models:
                raise Dwarf_Exception(f"config['image_color_model'] is not in {self.image_color_models}")
            if self.image_geometry is not None:
                if not isinstance(self.image_geometry, (list, tuple)) or len(self.image_geometry) != 2:
                    raise Dwarf_Exception("image_geometry must be None or a sequence of 2 ints")
                if not all(isinstance(x, int) and x > 0 for x in self.image_geometry):
                    raise Dwarf_Exception("image_geometry values must be positive ints")
            if self.stop_event is None or not isinstance(self.stop_event, MPEvent):
                raise Dwarf_Exception("config['stop_event'] is not a multiprocessing.Event")
            if not isinstance(self.exif_transpose, bool):
                raise Dwarf_Exception("config['exif_transpose'] is not a bool")
            if not isinstance(self.queue_free, MPQueue):
                raise Dwarf_Exception("config['queue_free'] is not a multiprocessing.Queue")
            if self.run_id is None:
                raise Dwarf_Exception("config['run_id'] is not ready")
            if self.MANIFEST_FILE_PATH is None:
                raise Dwarf_Exception("config['manifest_file_path'] is not ready")
            if self.database_path is None:
                raise Dwarf_Exception("config['database_path'] is not ready")
            self.db_conn = self.open_db(self.database_path)
            try:
                self.init_db()
            except Exception:
                self.db_conn.close()
                self.db_conn = None
                raise
        except Exception as e:
            raise Dwarf_Exception(f"Failed to check state: {e}") from e
        return True

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
            # match self.image_type:
            #     case np.float32:
            #         img_numpy = np.array(image, dtype=np.float32)
            #     case np.float64:
            #         img_numpy = np.array(image, dtype=np.float64)
            #     case np.int32:
            #         img_numpy = np.array(image, dtype=np.int32)
            #     case np.int64:
            #         img_numpy = np.array(image, dtype=np.int64)
            #     case np.uint32:
            #         img_numpy = np.array(image, dtype=np.uint32)
            #     case np.uint64:
            #         img_numpy = np.array(image, dtype=np.uint64)
            #     case _:
            #         raise Dwarf_Exception("Incorrect image type")
            img_numpy = np.array(image, dtype=self.image_type)
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
            resample_map = {
                "nearest": Image.Resampling.NEAREST,
                "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC,
                "lanczos": Image.Resampling.LANCZOS,
            }
            if self.image_geometry is not None:
                resample = resample_map[self.resample]
                image = image.resize(tuple(self.image_geometry), resample=resample)
            return image
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
            if self.exif_transpose:
                image = ImageOps.exif_transpose(image)
            image = self.image_to_correct_geometry(image)
            image = self.image_to_color_model(image)
            image = self.image_to_type(image)
            return image
        except Exception as e:
            raise Dwarf_Exception(f"Failed to format image: {e}") from e

    def put_ready_with_retry(self, handle: dict) -> bool:
        while not self.stop_event.is_set():
            try:
                self.queue_ready.put(handle, timeout=0.5)
                return True
            except Full:
                self.drain_free_queue()
                continue
            except (BrokenPipeError, OSError) as e:
                logger.error_with_context(e)
                self.queue_state = False
                return False
        return False

    def send_dead_pill(self, mode = "fast"):
        """
        Отправка сигнала о завершении сета картинок для эксперимента

        Raises:
            Dwarf_Exception: Ошибка
        """
        match mode:
            case "fast":
                try:
                    if not self.queue_state:
                        return
                    empty_since = None

                    while self.shm_refs and not self.stop_event.is_set():
                        self.drain_free_queue()
                        if not self.shm_refs:
                            break

                        now = time.monotonic()
                        if self.queue_ready.empty():
                            if empty_since is None:
                                empty_since = now
                            elif now - empty_since >= 30.0:
                                break
                        else:
                            empty_since = None

                        self.stop_event.wait(timeout=0.1)

                    self.drain_free_queue()
                    try:
                        self.queue_ready.put(DEAD_PILL, block=True, timeout=5)
                        logger.log("Sended dead pill to Queue")
                    except (BrokenPipeError, OSError) as e:
                        self.queue_state = False
                        logger.error_with_context(e)
                        return
                    except Exception as e:
                        logger.error_with_context(e)
                        return
                except Exception as e:
                    logger.error_with_context(e)
            case "long":
                try:
                    if not self.queue_state:
                        return
                    dead_pill_retry_count = 3
                    while dead_pill_retry_count > 0:
                        try:
                            self.queue_ready.put(None, block=True)
                            logger.log("Sended dead pill to Queue")
                            break
                        except (BrokenPipeError, OSError) as e:
                            self.queue_state = False
                            logger.error_with_context(e)
                            break
                        except Exception as e:
                            logger.error_with_context(e)
                            dead_pill_retry_count -= 1
                            time.sleep(10)
                            if dead_pill_retry_count == 0:
                                logger.log("Failed to send dead pill to Queue")
                                break
                except Exception as e:
                    logger.error_with_context(e)

    def save_to_shm(self, numpy_image: np.ndarray) -> dict:
        shm_name = make_shm_name(self.run_id, self.image_id)
        try:
            shm = shared_memory.SharedMemory(create=True, size=numpy_image.nbytes, name=shm_name)
        except FileExistsError:
            old = shared_memory.SharedMemory(name=shm_name)
            old.close()
            old.unlink()
            shm = shared_memory.SharedMemory(create=True, size=numpy_image.nbytes, name=shm_name)
        view = np.ndarray(numpy_image.shape, dtype=numpy_image.dtype, buffer=shm.buf)
        view[:] = numpy_image[:]
        self.shm_refs[self.image_id] = shm
        handle = {
            "image_id": self.image_id,
            "shm_name": shm_name,
            "shape": numpy_image.shape,
            "dtype": str(numpy_image.dtype),
            "nbytes": numpy_image.nbytes
        }
        return handle

    def drain_free_queue(self) -> None:
        while True:
            try:
                image_id = self.queue_free.get_nowait()
            except Empty:
                return
            shm = self.shm_refs.pop(image_id, None)
            if shm is None:
                continue
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                logger.error_with_context(e)

    def mark_done(self, image_id: str) -> None:
        self.db_conn.execute(
            "INSERT OR IGNORE INTO processed_images (image_id) VALUES (?)",
            (image_id,),
        )

    def check_done_picture(self) -> bool:
        cursor = self.db_conn.execute("SELECT 1 FROM processed_images WHERE image_id = ? LIMIT 1", (self.image_id,),)
        return cursor.fetchone() is not None

    def open_db(self, dp_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(
            dp_path,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def init_db(self) -> None:
        self.db_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_images (
                image_id TEXT PRIMARY KEY
            ) WITHOUT ROWID
            """
        )

    def main_loop(self):
        """
        Основной цикл

        Raises:
            Dwarf_Exception: Ошибка
        """
        try:
            if not self.check_state():
                raise Dwarf_Exception("State is incorrect, main loop is not running")
            with open(self.MANIFEST_FILE_PATH) as f:
                for line in f:
                    self.drain_free_queue()
                    if self.stop_event.is_set():
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        self.image_id = data["image_id"]
                        if  self.check_done_picture():
                            logger.log(f"Picture {data['image_id']} is already processed")
                            continue
                        try:
                            with Image.open(data["path"]) as raw_image:
                                numpy_image = self.formatter(raw_image)
                        except Exception as e:
                            logger.error_with_context(e)
                            continue
                        handle = self.save_to_shm(numpy_image)
                        if not self.put_ready_with_retry(handle):
                            break
                        else:
                            self.mark_done(self.image_id)
                    except Exception as e:
                        logger.error_with_context(e)
                        continue
        except Exception as e:
            logger.error_with_context(e)
        finally:
            try:
                self.send_dead_pill()
            except Exception as e:
                logger.error_with_context(e)
            try:
                if self.db_conn is not None:
                    self.db_conn.close()
            except Exception as e:
                logger.error_with_context(e)
