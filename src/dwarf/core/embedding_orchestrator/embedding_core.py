from src.dwarf.core.utils.utils import *

class Embedding_Core_Meta(abc.ABCMeta):
    def __getattr__(cls, name: str): 
        if name in cls._registered_embeddings:
            return cls._registered_embeddings[name]
        raise AttributeError(f"{cls.__name__} has no attribute {name}")

class Embedding_Core(abc.ABC, metaclass=Embedding_Core_Meta):
    """
    Класс встраивания.
    От него наследуются все типы встраиваний. От каждого типа встраиваний наследуются классы конкретных встраиваний.

    Attributes:
        _registered_embeddings (dict): Словарь зарегистрированных встраиваний.

    Methods:
        __init_subclass__(cls, **kwargs): 
            Регистрирует встраивание в словаре _registered_embeddings.
        get_registered_embeddings(): 
            Возвращает словарь _registered_embeddings.
        get_all_embeddings(cls): 
            Возвращает все встраивания, наследуемые от cls.
        embedding(args: dict = {
            "input_data": None,
            "output_data": None
        }): 
            Абстрактный метод встраивания, который должен быть реализован в каждом подклассе.
        extraction(args: dict = {
            "input_data": None,
            "output_data": None
        }): 
            Абстрактный метод извлечения, который должен быть реализован в каждом подклассе.
    
    """
    _registered_embeddings = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Embedding_Core._registered_embeddings[cls.__name__] = cls

    def get_registered_embeddings():
        return Embedding_Core._registered_embeddings

    @classmethod
    def get_all_embeddings(cls):
        return cls.__subclasses__()

    @staticmethod
    @abc.abstractmethod
    def embedding(args: dict = {
        "input_data": None,
        "output_data": None
    }): 
        pass

    @staticmethod
    @abc.abstractmethod
    def extraction(args: dict = {
        "input_data": None,
        "output_data": None
    }): 
        pass

    def get_embedding_class_by_name(embedding_name: str):
        return Embedding_Core._registered_embeddings[embedding_name]

    def use_embeddings(embeddings: dict):
        all_embeddings = embeddings.keys()
        for embedding_name in all_embeddings:
            Embedding_Core.get_embedding_class_by_name(embedding_name).embedding(embeddings[embedding_name])

class Ready_Spatial_Embeddings(Embedding_Core):
    """
    Класс готовых решений для пространственного встраивания.
    """

class Ready_Frequency_Embeddings(Embedding_Core):
    """
    Класс с готовыми частотными внедрениями.
    """

class Ready_Quantization_Based_Embeddings(Embedding_Core):
    """
    Класс с готовыми внедрениями на основе квантования.
    """

class Ready_Spread_Spectrum_Embeddings(Embedding_Core):
    """
    Класс с готовыми спектральными внедрениями.
    """

class Ready_SOTA_Embeddings(Embedding_Core):
    """
    Класс с готовыми SOTA внедрениями.
    """