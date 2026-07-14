from ..utils.utils import import_function

class AttackCore:
    def __init__(self):
        self.available_attacks: dict = {}
        pass

    def get_new_attack(self, attacks: dict):

        for attack in attacks:
            # self.available_attacks[attack[attack.rfind("\\", 0, len(attack)) + 1:]] = import_function()
            self.available_attacks[attack] = import_function(attack, attacks[attack])
        return
    
    def get_all_available_attacks(self):
        for attack in self.available_attacks:
            print(attack)
        return
