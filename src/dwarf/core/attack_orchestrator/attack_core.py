from src.dwarf.core.utils.utils import *

if TYPE_CHECKING:
    from src.dwarf.ready_solutions.attack_solutions.geometric_attacks import Ready_Geometric_Attacks

class Attack_Core:

    if TYPE_CHECKING:
        geometric_attacks: Ready_Geometric_Attacks
    def __init__(self):
        self.available_attacks: dict[str, Any] = {}

        self.ready_attack_types: dict = {
            "geometric_attacks": str(DWARF_DIR) + r"\ready_solutions\attack_solutions\geometric_attacks.py",
        }
        pass

    def __getattr__(self, attack_type: str):
        if attack_type in self.available_attacks:
            return self.available_attacks[attack_type]
        else:
            raise AttributeError

    def get_new_attack(self, attacks: dict):

        for attack in attacks:
            # self.available_attacks[attack[attack.rfind("\\", 0, len(attack)) + 1:]] = import_function()
            self.available_attacks[attack] = import_function(attack, attacks[attack])
        return
    
    def get_all_available_attacks(self):
        for attack in self.available_attacks:
            print(attack)
        return
    
    def connect_ready_attacks(self, attacks_type: list = [], connect_all: bool = False):
        for attack in attacks_type:
            self.available_attacks[attack] = import_function(attack, self.ready_attack_types[attack])

        if connect_all:
            for attack in self.ready_attack_types:
                self.available_attacks[attack] = import_function(attack, self.ready_attack_types[attack])
        return
