import atexit
import inspect
import logging
import os
import time
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from sys import exc_info
from zoneinfo import ZoneInfo


def get_local_time(timestamp):
    """Конвертер для logging.Formatter, возвращает struct_time в Москве."""
    dt = datetime.fromtimestamp(timestamp, tz=ZoneInfo('Europe/Moscow'))
    return dt.timetuple()


DEFAULT_FORMAT = '%(asctime)s | %(levelname)s | %(message)s'
FAILOVER_FORMAT = '%(asctime)s | %(message)s'


class ChannelFilter(logging.Filter):
    """Пропускает только записи, у которых extra['channel'] совпадает."""

    def __init__(self, channel):
        super().__init__()
        self.channel = channel

    def filter(self, record):
        return getattr(record, "channel", "default") == self.channel


class SimpleLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name="APP", channels=None,
                 max_bytes=100 * 1024 * 1024, backup_count=3):
        if getattr(self, "_initialized", False):
            return

        if channels is None:
            channels = {
                "default": "logs.txt",
                "failover": "failover.log",
            }
        self.channels = channels

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self._handlers = {}
        self._last_log = {}

        default_fmt = logging.Formatter(DEFAULT_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
        default_fmt.converter = get_local_time

        failover_fmt = logging.Formatter(FAILOVER_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
        failover_fmt.converter = get_local_time

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(default_fmt)
        console_handler.addFilter(ChannelFilter("default"))
        self.logger.addHandler(console_handler)
        self._handlers["console"] = console_handler

        for channel, filepath in channels.items():
            formatter = failover_fmt if channel == "failover" else default_fmt
            try:
                fh = RotatingFileHandler(
                    filepath,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                fh.setFormatter(formatter)
                fh.addFilter(ChannelFilter(channel))
                self.logger.addHandler(fh)
                self._handlers[channel] = fh
            except Exception as e:
                print(f"Error of creation handler-а for channel '{channel}': {e}")

        atexit.register(self.cleanup)
        self._initialized = True

    def cleanup(self):
        for handler in list(self.logger.handlers):
            try:
                handler.close()
                self.logger.removeHandler(handler)
            except Exception:
                pass
        self._handlers.clear()

    def log(self, message, channel="default", level=logging.INFO):
        """Пишет сообщение в указанный канал. По умолчанию — 'default'."""
        if self._last_log.get(channel) != message:
            self.logger.log(level, message, extra={"channel": channel})
            self._last_log[channel] = message

    def failover(self, name, reason):
        """Пишет запись о провалившейся фотке в канал 'failover'.
        Формат: '<name> | <reason>'.
        Дедупликация не применяется — каждая запись важна."""
        msg = f"{name} | {reason}"
        self.logger.info(msg, extra={"channel": "failover"})

    def error_with_context(self, error):
        exc_type, exc_value, exc_tb = exc_info()
        if exc_tb is not None:
            while exc_tb.tb_next:
                exc_tb = exc_tb.tb_next
            filename = exc_tb.tb_frame.f_code.co_filename
            line_no = exc_tb.tb_lineno
        else:
            frame = inspect.currentframe().f_back
            filename = frame.f_code.co_filename
            line_no = frame.f_lineno

        filename = os.path.basename(filename)
        self.logger.error(
            f"ERROR: {str(error)} | File: {filename} | Line: {line_no}",
            extra={"channel": "default"}
        )

    def timeit(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            self.log(f"Start '{func.__name__}'")
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                self.log(f"Finished '{func.__name__}' in {elapsed:.2f}s")
                return result
            except Exception as e:
                self.error_with_context(e)
                raise
        return wrapper


logger = SimpleLogger()
