from typing import Any


class Colors():
    """ANSI codes for coloring and formatting console output."""
    HEADER: str = '\033[95m'
    OKBLUE: str = '\033[94m'
    OKCYAN: str = '\033[96m'
    OKGREEN: str = '\033[92m'
    WARNING: str = '\033[93m'
    FAIL: str = '\033[91m'
    ENDC: str = '\033[0m'
    BOLD: str = '\033[1m'
    UNDERLINE: str = '\033[4m'


class ParsingError(Exception):
    def __init__(self, type_name: str) -> None:
        self.bold: str = Colors.BOLD
        self.warning: str = Colors.WARNING
        self.fail: str = Colors.FAIL
        self.end: str = Colors.ENDC
        self.name: str = type_name

    def __str__(self) -> Any:
        color_start = self.bold + self.fail
        color_end = self.end + self.bold
        return color_start + f'{self.name}' + color_end + ":"


class InvalidJSON(ParsingError):
    name = "InvalidJSON"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(InvalidJSON.name)

    def __str__(self) -> Any:
        error = super().__str__()
        return error + f' {self.message}'


class NoPermissionErr(ParsingError):
    name = "PermissionError"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(NoPermissionErr.name)

    def __str__(self) -> Any:
        error = super().__str__()
        return error + self.message


class FileNotFoundErr(ParsingError):
    name = "PermissionError"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(FileNotFoundErr.name)

    def __str__(self) -> Any:
        error = super().__str__()
        return error + self.message


class InvalidParameterValue(ParsingError):
    name = "InvalidParameterValue"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(InvalidParameterValue.name)

    def __str__(self) -> Any:
        error = super().__str__()
        return error + self.message
