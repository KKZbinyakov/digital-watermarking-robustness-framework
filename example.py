# Это файл, в котором будут рекомендации по написанию кода.

def name_of_function(names_of_variables):
    x, word, words_with_space = names_of_variables
    return

class MyClass:
    def __init__(self, names_of_variables):
        x, word, words_with_space = names_of_variables # перменные называем так

    def _private_method(self, names_of_variables: list, something_else: bool) -> None:
        """
        Это комментарии к методу. Пишем их для любой функции или класса, поскольку кодом пользуются несколько человек, это необходимо для удобной работы.

        Args:
            names_of_variables (list): описание переменных
            something_else (bool): также для каждого аргумента и самой функции даём не только описание, но и строгий тип данных, для уменьшения путаницы и лучшего контроля кода.

        Returns:
            None
        """
        x, word, words_with_space = names_of_variables

# Директрии utils нужны для функций, которые напрямую не относятся к тем частям кода, в коротых они лежат, но при этом нужны для работы. Например, в core части 
# в utils убдет лежать функция открытия файла. Она нужна для функционала core, но при этом не является его смысловым ядром. Также важно, функции, которые понадобятся и в 
# core, и в utils, должны быть в одной директории - common_utils, чтобы избежать дублирования.
