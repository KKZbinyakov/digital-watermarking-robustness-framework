from src.dwarf.core.utils.utils import *

class ExpertiseCore:
    def __init__(self):
        self.available_expertise: dict = {}
        pass

    def get_new_expertise(self, expertises: dict):

        for expertise in expertises:
            # self.available_expertise[expertise[expertise.rfind("\\", 0, len(expertise)) + 1:]] = import_function()
            self.available_expertise[expertise] = import_function(expertise, expertises[expertise])
        return
    
    def get_all_available_expertise(self):
        for expertise in self.available_expertise:
            print(expertise)
        return