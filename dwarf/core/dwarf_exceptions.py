class Dwarf_Exception(Exception):
    """
    Класс исключений Dwarf

    Args:
        message (str, optional): Описание исключения.

    Returns:
        str: Описание исключения
    """
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"Dwarf Exception: {self.message}"
        else:
            return "Dwarf Exception has been raised"
