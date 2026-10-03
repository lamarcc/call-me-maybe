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


class ProgramError(Exception):
    """Base class of every custom error of the program.

    Subclasses only override the class attribute `name`, which is the
    label printed before the message.

    Attributes:
    name -- label of the error, read from the most derived class
    message -- explanation of what went wrong
    """

    name: str = "ProgramError"

    def __init__(self, message: str) -> None:
        """Store the message that explains the error."""
        super().__init__(message)
        self.message = message

    def __str__(self) -> Any:
        """Return '<name>: <message>' with ANSI colors."""
        color_start = Colors.BOLD + Colors.FAIL
        color_end = Colors.ENDC + Colors.BOLD
        return (
            color_start + self.name + color_end + ": "
            + self.message + Colors.ENDC
        )

    @staticmethod
    def msg(message: str) -> str:
        """Return a message with ANSI colors."""
        color_start = Colors.BOLD + Colors.WARNING
        color_end = Colors.ENDC + Colors.BOLD
        new_line = '\n'
        return (
            color_start + new_line + message + color_end + Colors.ENDC
        )


class ParsingError(ProgramError):
    """Base class of errors that make an input file unusable.

    The program cannot run at all and exits with status 1.
    """

    name = "ParsingError"


class InvalidJSON(ParsingError):
    """Raise when a file is not valid JSON or is not a JSON list."""

    name = "InvalidJSON"


class NoPermissionErr(ParsingError):
    """Raise when an input file cannot be read because of permissions."""

    name = "PermissionError"


class FileNotFoundErr(ParsingError):
    """Raise when an input path does not exist or is a directory."""

    name = "FileNotFoundError"


class InvalidFunctionDefinition(ParsingError):
    """Raise when a function definition is missing or malformed."""

    name = "InvalidFunctionDefinition"


class InvalidPrompt(ParsingError):
    """Raise when a prompt entry has no non-empty "prompt" string."""

    name = "InvalidPrompt"


class InvalidParameterValue(ParsingError):
    """Raise when a parameter has a missing or unsupported type."""

    name = "InvalidParameterValue"


class GenerationError(ProgramError):
    """Base class of errors raised while generating one prompt.

    Only the current prompt fails; the other prompts still run.
    """

    name = "GenerationError"


class FunctionNotFound(GenerationError):
    """Raise when the model does not produce a known function name."""

    name = "FunctionNotFound"


class InvalidGeneratedValue(GenerationError):
    """Raise when a generated value is unclosed or has the wrong type."""

    name = "InvalidGeneratedValue"


class DirCreationError(ProgramError):
    """Raise when the directory already exist while trying creating it"""

    name = "DirCreationError"
