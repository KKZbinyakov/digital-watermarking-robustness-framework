from src.dwarf.core.utils.utils import *

class Attack_Core_Meta(abc.ABCMeta):
    def __getattr__(cls, name: str): 
        if name in cls._registered_attacks:
            return cls._registered_attacks[name]
        raise AttributeError(f"{cls.__name__} has no attribute {name}")

class Attack_Core(abc.ABC, metaclass=Attack_Core_Meta):
    """
    Класс-оркестратор атак.
    От него наследуются все типы атак. От каждого типа атак наследуются классы конкретных атак.

    Attributes:
        _registered_attacks (dict): Словарь, в котором хранятся все зарегистрированные атаки.

    Methods:
        __init_subclass__(cls, **kwargs): 
            Регистрирует атаку в словаре _registered_attacks.
        get_registered_attacks(): 
            Возвращает словарь _registered_attacks.
        get_all_attacks(cls): 
            Возвращает все атаки, наследуемые от cls.
        attack(args: dict = {
            "input_data": None,
            "output_data": None
        }): 
            Абстрактный метод атак, который должен быть реализован в каждом подклассе.

    """
    _registered_attacks = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Attack_Core._registered_attacks[cls.__name__] = cls

    def get_registered_attacks():
        return Attack_Core._registered_attacks

    @classmethod
    def get_all_attacks(cls):
        return cls.__subclasses__()

    @staticmethod
    @abc.abstractmethod
    def attack(args: dict = {
        "input_data": None,
        "output_data": None
    }): 
        pass

    def get_attack_class_by_name(attack_name: str):
        return Attack_Core._registered_attacks[attack_name]

    def use_attacks(attacks: dict):
        all_attacks = attacks.keys()
        for attack_name in all_attacks:
            Attack_Core.get_attack_class_by_name(attack_name).attack(attacks[attack_name])

class Ready_Geometric_Attacks(Attack_Core):
    """
    Класс с готовыми геометрическими атаками.
    """

class Ready_Noise_Attacks(Attack_Core):
    """
    Класс с готовыми шумовыми атаками.
    """

class Ready_Filtering_Attacks(Attack_Core):
    """
    Класс с готовыми фильтрационными атаками.
    """

class Ready_Compression_Attacks(Attack_Core):
    """
    Класс с готовыми сжатием атаками.
    """

class Ready_Color_Brightness_Attacks(Attack_Core):
    """
    Класс с готовыми атаками изменения яркости.
    """

class Ready_SOTA_Attacks(Attack_Core):
    """
    Класс с готовыми SOTA атаками.
    """

class Ready_Watermark_Removal_Networks_Attacks(Attack_Core):
    """
    Класс с готовыми атаками Watermark Removal Networks.
    """

class Ready_Adversarial_Examples_Attacks(Attack_Core):
    """
    Класс с готовыми адверсариальными примерами.
    """

class Ready_Collage_Synchronization_Attacks(Attack_Core):
    """
    Класс с готовыми атаками синхронизации коллажа.
    """

class Ready_Physical_Attacks(Attack_Core):
    """
    Класс с готовыми физическими атаками.
    """
