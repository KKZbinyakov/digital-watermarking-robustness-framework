from src.dwarf.core.utils.utils import *

class Expertise_Core_Meta(abc.ABCMeta):
    def __getattr__(cls, name: str): 
        if name in cls._registered_expertises:
            return cls._registered_expertises[name]
        raise AttributeError(f"{cls.__name__} has no attribute {name}")

class Expertise_Core(abc.ABC, metaclass=Expertise_Core_Meta):
    """
    Класс, содержащий в себе все готовые решения для экспертизы.

    Attributes:
    _registered_expertises : dict
        Словарь, содержащий в себе все готовые решения для экспертизы.

    Methods:
    __init_subclass__(cls, **kwargs) 
        Регистрирует экспертизу в словаре _registered_expertises.
    get_registered_expertises()
        Возвращает словарь _registered_expertises.
    get_all_expertises(cls)
        Возвращает все экспертизы, наследуемые от cls.
    expertise(args: dict = {
        "original_bits": None,
        "extracted_bits": None
    }): 
        Абстрактный метод экспертизы, который должен быть реализован в каждом подклассе.

    """
    _registered_expertises = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Expertise_Core._registered_expertises[cls.__name__] = cls

    def get_registered_expertises():
        return Expertise_Core._registered_expertises

    @classmethod
    def get_all_expertises(cls):
        return cls.__subclasses__()

    @staticmethod
    @abc.abstractmethod
    def expertise(args: dict = {
        "original_bits": None,
        "extracted_bits": None
    }): 
        pass

    def get_expertise_class_by_name(expertise_name: str):
        return Expertise_Core._registered_expertises[expertise_name]

    def use_expertises(expertises: dict):
        all_expertises = expertises.keys()
        for expertise_name in all_expertises:
            Expertise_Core.get_expertise_class_by_name(expertise_name).expertise(expertises[expertise_name])

class Ready_Robustness_Expertise(Expertise_Core):
    """
    Класс, содержащий в себе все готовые решения для экспертизы робастности.
    """
class Ready_Imperceptibility_Expertise(Expertise_Core):
    """
    Класс, содержащий в себе все готовые решения для экспертизы невидимости.
    """