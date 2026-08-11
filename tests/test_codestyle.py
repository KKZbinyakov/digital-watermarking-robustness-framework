"""Машинная проверка правил написания кода.

Правила из Codestyle Bible проверяются статически, по исходникам: разбор `.py`
идёт через ast, `.pyx` — построчно, поскольку Cython ast не разбирает. Смысл в
том, чтобы нарушение ловилось на CI, а не на ревью.
"""

import ast
import re
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parent.parent / "dwarf"
PYTHON_SOURCES = sorted(PACKAGE_ROOT.rglob("*.py"))
CYTHON_SOURCES = sorted(PACKAGE_ROOT.rglob("*.pyx"))
ALL_SOURCES = PYTHON_SOURCES + CYTHON_SOURCES

ENTRY_POINTS = ("attack", "expertise", "embedding", "extraction", "ds")
"""Методы решений, к которым относятся правила о defaults и об аргументах."""

CYRILLIC = re.compile(r"[А-Яа-яЁё]")
KNOWN_VIOLATIONS = {
    "warnings_are_english": {
        "dwarf/core/utils/utils.py",
        "dwarf/ready_solutions/embedding_solutions/spatial/lsb.py",
    },
    "no_relative_imports": {
        "dwarf/__init__.py",
        "dwarf/core/__init__.py",
    },
    "defaults_cover_every_used_key": {
        "dwarf/ready_solutions/embedding_solutions/spatial/lsb.py",
    },
    "solutions_do_not_take_paths": {
        "dwarf/ready_solutions/embedding_solutions/spatial/lsb.py",
    },
}
"""Известные нарушения, лежащие вне текущей зоны работ.
Файлы ядра правилам пока не соответствуют, но их правка — отдельная задача;
lsb.py ждёт перевода встраиваний на матричный контракт. Долг записан пофайлово
и попроверочно, помечен xfail(strict=True) и потому виден в отчёте CI. Когда
файл починят, strict превратит неожиданный успех в падение и напомнит убрать
запись отсюда.
"""


def relative(path):
    """Путь относительно корня репозитория — так сообщения об ошибках читаемее."""
    return str(path.relative_to(PACKAGE_ROOT.parent))


def sources(check, paths):
    """
    Готовит параметры проверки, помечая известные нарушения как ожидаемые.

    Args:
        check (str): имя проверки, ключ в KNOWN_VIOLATIONS
        paths (list): список файлов, к которым проверка применяется
    Returns:
        list: параметры pytest, часть из них с меткой xfail
    """
    pending = KNOWN_VIOLATIONS.get(check, set())
    parameters = []
    for path in paths:
        name = relative(path).replace("\\", "/")
        marks = [
            pytest.mark.xfail(
                strict=True,
                reason="вне зоны текущих работ",
            )
        ] if name in pending else []
        parameters.append(pytest.param(path, marks=marks, id=name))
    return parameters


def parsed(path):
    """Разбирает модуль Python в дерево ast."""
    return ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )


def raise_blocks(path):
    """
    Возвращает текст всех операторов raise построчно, включая многострочные.

    Используется для `.pyx`: ast их не разбирает, а сообщения проверить нужно.

    Args:
        path (Path): путь к файлу

    Returns:
        list: список строк, каждая — текст одного оператора raise
    """
    blocks = []
    depth = 0
    current = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not current and line.strip().startswith("raise "):
            current = [line]
            depth = line.count("(") - line.count(")")

            if depth <= 0:
                blocks.append("\n".join(current))
                current = []

            continue

        if current:
            current.append(line)
            depth += line.count("(") - line.count(")")

            if depth <= 0:
                blocks.append("\n".join(current))
                current = []

    return blocks


def defaults_keys(node):
    """
    Собирает ключи словаря defaults внутри функции.

    Args:
        node (ast.FunctionDef): узел функции

    Returns:
        set | None: множество ключей либо None, если defaults не объявлен
    """
    for statement in ast.walk(node):
        if isinstance(statement, ast.Assign):
            targets = [
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            ]

            if (
                "defaults" in targets
                and isinstance(statement.value, ast.Dict)
            ):
                return {
                    key.value
                    for key in statement.value.keys
                    if isinstance(key, ast.Constant)
                }

    return None


def args_keys(node):
    """
    Собирает ключи, читаемые из args по индексу.

    Args:
        node (ast.FunctionDef): узел функции

    Returns:
        set: множество имён ключей
    """
    keys = set()

    for statement in ast.walk(node):
        if (
            isinstance(statement, ast.Subscript)
            and isinstance(statement.value, ast.Name)
            and statement.value.id == "args"
            and isinstance(statement.slice, ast.Constant)
        ):
            keys.add(statement.slice.value)

    return keys


def entry_functions(tree):
    """Возвращает пары (класс, функция) для точек входа решений."""
    found = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name in ENTRY_POINTS
                ):
                    found.append((node.name, item))

    return found


def _indent_width(line):
    """Возвращает ширину начального отступа строки с учётом табов."""
    prefix = line[:len(line) - len(line.lstrip(" \t"))]
    return len(prefix.expandtabs(4))


def cython_entry_functions(path):
    """
    Возвращает entry-point методы Cython вместе с их исходным текстом.

    Результат состоит из троек:
        (имя класса, имя функции, исходный текст функции)

    Cython-файлы не разбираются через стандартный ast, поэтому границы
    классов и функций определяются по отступам.

    Поддерживаются:
        class
        cdef class
        def
        cpdef

    Args:
        path (Path): путь к .pyx-файлу

    Returns:
        list: список троек
            (имя класса, имя функции, исходный текст функции)
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    class_pattern = re.compile(
        r"^(?P<indent>[ \t]*)"
        r"(?:cdef\s+)?"
        r"class\s+"
        r"(?P<name>[A-Za-z_]\w*)\b"
    )

    entry_pattern = re.compile(
        rf"^(?P<indent>[ \t]*)"
        rf"(?:def|cpdef)\s+"
        rf"(?P<name>{'|'.join(map(re.escape, ENTRY_POINTS))})"
        rf"\s*\("
    )

    classes = []
    found = []

    for index, line in enumerate(lines):
        if not line.strip():
            continue

        indent = _indent_width(line)

        # Если текущая строка находится на уровне класса или выше,
        # значит предыдущий класс закончился.
        while classes and indent <= classes[-1][0]:
            classes.pop()

        class_match = class_pattern.match(line)

        if class_match:
            classes.append(
                (
                    indent,
                    class_match.group("name"),
                )
            )
            continue

        entry_match = entry_pattern.match(line)

        if not entry_match or not classes:
            continue

        function_name = entry_match.group("name")
        function_indent = indent
        class_name = classes[-1][1]

        block = [line]

        # Функция продолжается, пока следующие непустые строки имеют
        # больший отступ, чем строка def/cpdef.
        for candidate in lines[index + 1:]:
            if (
                candidate.strip()
                and _indent_width(candidate) <= function_indent
            ):
                break

            block.append(candidate)

        found.append(
            (
                class_name,
                function_name,
                "\n".join(block),
            )
        )

    return found


def cython_defaults_keys(source):
    """
    Собирает ключи defaults внутри одной Cython entry-point функции.

    Args:
        source (str): исходный текст одной функции

    Returns:
        set | None: множество ключей либо None,
            если defaults не объявлен
    """
    block = re.search(
        r"\bdefaults\s*=\s*\{(.*?)\}",
        source,
        re.S,
    )

    if block is None:
        return None

    return set(
        re.findall(
            r"""["']([^"']+)["']\s*:""",
            block.group(1),
        )
    )


def cython_args_keys(source):
    """
    Собирает ключи, читаемые через args[...] внутри одной Cython-функции.

    Args:
        source (str): исходный текст одной функции

    Returns:
        set: множество имён ключей
    """
    return set(
        re.findall(
            r"""args\s*\[\s*["']([^"']+)["']\s*\]""",
            source,
        )
    )


# --- правило 1: вывод в терминал на английском ----------------------------


@pytest.mark.parametrize(
    "path",
    PYTHON_SOURCES,
    ids=relative,
)
def test_python_exception_messages_are_english(path):
    for node in ast.walk(parsed(path)):
        if not isinstance(node, ast.Raise):
            continue

        for text in ast.walk(node):
            if (
                isinstance(text, ast.Constant)
                and isinstance(text.value, str)
            ):
                assert not CYRILLIC.search(text.value), (
                    f"{relative(path)}: русский текст "
                    f"в исключении — {text.value!r}"
                )


@pytest.mark.parametrize(
    "path",
    CYTHON_SOURCES,
    ids=relative,
)
def test_cython_exception_messages_are_english(path):
    for block in raise_blocks(path):
        assert not CYRILLIC.search(block), (
            f"{relative(path)}: русский текст в исключении —\n"
            f"{block}"
        )


@pytest.mark.parametrize(
    "path",
    sources(
        "warnings_are_english",
        PYTHON_SOURCES,
    ),
)
def test_warnings_are_english(path):
    for node in ast.walk(parsed(path)):
        if not isinstance(node, ast.Call):
            continue

        target = node.func

        name = (
            target.attr
            if isinstance(target, ast.Attribute)
            else getattr(target, "id", "")
        )

        if name not in ("warn", "print"):
            continue

        for text in ast.walk(node):
            if (
                isinstance(text, ast.Constant)
                and isinstance(text.value, str)
            ):
                assert not CYRILLIC.search(text.value), (
                    f"{relative(path)}: русский текст "
                    f"в выводе — {text.value!r}"
                )


# --- правило 2: дефолт на каждую используемую переменную ------------------


@pytest.mark.parametrize(
    "path",
    PYTHON_SOURCES,
    ids=relative,
)
def test_solutions_declare_defaults(path):
    for class_name, function in entry_functions(parsed(path)):
        if not args_keys(function):
            continue

        assert defaults_keys(function) is not None, (
            f"{relative(path)}: "
            f"{class_name}.{function.name} "
            f"не объявляет defaults"
        )


@pytest.mark.parametrize(
    "path",
    sources(
        "defaults_cover_every_used_key",
        PYTHON_SOURCES,
    ),
)
def test_defaults_cover_every_used_key(path):
    for class_name, function in entry_functions(parsed(path)):
        declared = defaults_keys(function)

        if declared is None:
            continue

        missing = args_keys(function) - declared

        assert not missing, (
            f"{relative(path)}: "
            f"{class_name}.{function.name} "
            f"читает {sorted(missing)} без дефолта"
        )


@pytest.mark.parametrize(
    "path",
    CYTHON_SOURCES,
    ids=relative,
)
def test_cython_defaults_cover_every_used_key(path):
    """
    Проверяет defaults отдельно для каждого Cython entry-point.

    В отличие от старой реализации, args[...] и defaults не собираются
    по всему .pyx-файлу сразу. Каждый embedding/extraction/attack/etc.
    сверяется только со своим собственным словарём defaults.
    """
    for (
        class_name,
        function_name,
        source,
    ) in cython_entry_functions(path):

        used = cython_args_keys(source)

        if not used:
            continue

        declared = cython_defaults_keys(source)

        assert declared is not None, (
            f"{relative(path)}: "
            f"{class_name}.{function_name} "
            f"не объявляет defaults"
        )

        missing = used - declared

        assert not missing, (
            f"{relative(path)}: "
            f"{class_name}.{function_name} "
            f"читает {sorted(missing)} без дефолта"
        )


@pytest.mark.parametrize(
    "path",
    ALL_SOURCES,
    ids=relative,
)
def test_no_args_get_fallbacks(path):
    """
    Дефолт задаётся один раз, в defaults:
    args.get дублировал бы его.
    """
    source = path.read_text(encoding="utf-8")

    assert "args.get(" not in source, (
        f"{relative(path)}: "
        f"дефолт продублирован через args.get"
    )


# --- правила 3 и 4: матрицы на входе и выходе решений ---------------------


@pytest.mark.parametrize(
    "path",
    sources(
        "solutions_do_not_take_paths",
        ALL_SOURCES,
    ),
)
def test_solutions_do_not_take_paths(path):
    """
    Решения принимают матрицы, а не пути:
    путь означает, что модуль сам лезет на диск.
    """
    if "ready_solutions" not in str(path):
        return

    source = path.read_text(encoding="utf-8")

    offenders = set(
        re.findall(
            r'args\["(\w*(?:path|data)\w*)"\]',
            source,
        )
    )

    assert not offenders, (
        f"{relative(path)}: "
        f"решение читает {sorted(offenders)} "
        f"вместо матрицы"
    )


@pytest.mark.parametrize(
    "path",
    ALL_SOURCES,
    ids=relative,
)
def test_entry_points_use_keyword_arguments(path):
    """
    Сигнатура вида attack(args: dict = {...}) —
    старый контракт и мутабельный дефолт.
    """
    source = path.read_text(encoding="utf-8")

    for entry in ENTRY_POINTS:
        assert f"def {entry}(args" not in source, (
            f"{relative(path)}: "
            f"{entry} объявлен со словарём вместо **args"
        )


# --- правило 6: формат импортов -------------------------------------------


@pytest.mark.parametrize(
    "path",
    ALL_SOURCES,
    ids=relative,
)
def test_no_star_imports(path):
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        code = line.split("#", 1)[0].strip()

        if (
            code.startswith(("from ", "import "))
            and code.endswith("import *")
        ):
            pytest.fail(
                f"{relative(path)}:{number}: "
                f"звёздный импорт — {code}"
            )


@pytest.mark.parametrize(
    "path",
    sources(
        "no_relative_imports",
        ALL_SOURCES,
    ),
)
def test_no_relative_imports(path):
    """
    Импорт указывается полным путём от корня пакета;
    cimport — синтаксис Cython, его не трогаем.
    """
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()

        if (
            stripped.startswith("from .")
            and "cimport" not in stripped
        ):
            pytest.fail(
                f"{relative(path)}:{number}: "
                f"относительный импорт — {stripped}"
            )