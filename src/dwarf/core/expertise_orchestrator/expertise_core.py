from src.dwarf.core.utils.utils import *

if TYPE_CHECKING:
    from src.dwarf.ready_solutions.expertise_solutions.robustness_solutions import Ready_Robustness_Expertise


class Expertise_Core:

    if TYPE_CHECKING:
        robustness_expertises: Ready_Robustness_Expertise

    def __init__(self):
        self.available_expertise: dict = {}
        self.ready_expertise_types: dict = {
            "imperceptibility_expertises": str(DWARF_DIR) + r"\ready_solutions\expertise_solutions\imperceptibility_solutions.py",
            "robustness_expertises": str(DWARF_DIR) + r"\ready_solutions\expertise_solutions\robustness_solutions.py",
        }
        pass

    def __getattr__(self, expertise_type: str):
        if expertise_type in self.available_expertise:
            return self.available_expertise[expertise_type]
        else:
            raise AttributeError

    def get_new_expertise(self, expertises: dict):

        for expertise in expertises:
            # self.available_expertise[expertise[expertise.rfind("\\", 0, len(expertise)) + 1:]] = import_function()
            self.available_expertise[expertise] = import_function(expertise, expertises[expertise])
        return
    
    def get_all_available_expertise(self):
        for expertise in self.available_expertise:
            print(expertise)
        return
    
    def connect_ready_expertises(self, expertises_type: list = [], connect_all: bool = False):
        for expertise in expertises_type:
            self.available_expertise[expertise] = import_function(expertise, self.ready_expertise_types[expertise])

        if connect_all:
            for expertise in self.ready_expertise_types:
                self.available_expertise[expertise] = import_function(expertise, self.ready_expertise_types[expertise])
        return