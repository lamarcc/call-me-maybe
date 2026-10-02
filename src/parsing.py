from __future__ import annotations
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Optional
import json
import errors


ALLOWED_TYPES = ("number", "integer", "float", "string", "boolean")


class Prompt(BaseModel):
    """One user request, read from the prompts file."""

    prompt: str = Field(min_length=1)


class Function(BaseModel):
    """One callable function, read from the function definitions file.

    Attributes:
    name -- unique name the model must generate
    description -- text shown to the model to choose the function
    parameters -- mapping of parameter name to {"type": <type>}
    nb_parameters -- number of parameters, at least 1
    returns -- type of the returned value, as {"type": <type>}
    """

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, Any]
    nb_parameters: int = Field(ge=1)
    returns: dict[str, str]


class Parse():
    """Load and validate the two input files.

    Every problem raises a ParsingError, so the generator only receives
    valid data.
    """

    def __init__(self, f_path: Optional[str], p_path: Optional[str]) -> None:
        """Store the input paths and prepare the empty results.

        A path that is None or empty falls back to its default file in
        data/input/.
        """
        if p_path:
            self.prompt_path: str = p_path
        else:
            self.prompt_path = "data/input/function_calling_tests.json"
        if f_path:
            self.function_path: str = f_path
        else:
            self.function_path = "data/input/functions_definition.json"
        self.all_functions: dict = {}
        self.all_prompts: list = []

    def _load_json(self, path: str) -> list:
        """Read a JSON file and return its top-level list.

        Raise InvalidJSON, NoPermissionErr or FileNotFoundErr when the
        file cannot be read or does not contain a JSON list.
        """
        try:
            with open(path, "r", encoding="utf-8") as file:
                content = json.load(file)
        except json.JSONDecodeError as e:
            raise errors.InvalidJSON(f"{path}: {e}")
        except PermissionError:
            raise errors.NoPermissionErr(f"{path}: permission denied")
        except FileNotFoundError:
            raise errors.FileNotFoundErr(f"{path}: the file does not exist")
        except IsADirectoryError:
            raise errors.FileNotFoundErr(f"{path}: is a directory")
        if not isinstance(content, list):
            raise errors.InvalidJSON(f"{path}: top level must be a list")
        return content

    def prompt(self) -> None:
        """Fill all_prompts from the prompts file.

        Raise InvalidPrompt when an entry has no non-empty "prompt" string.
        """
        for i, prompts in enumerate(self._load_json(self.prompt_path)):
            try:
                self.all_prompts.append(Prompt(prompt=prompts["prompt"]))
            except (KeyError, TypeError, ValidationError):
                raise errors.InvalidPrompt(
                    f'{self.prompt_path}: entry {i} needs a non-empty '
                    '"prompt" string'
                )

    def function(self) -> None:
        """Fill all_functions from the function definitions file.

        Raise InvalidFunctionDefinition when an entry is malformed, a name
        is defined twice or the file has no function. Raise
        InvalidParameterValue when a parameter type is not in
        ALLOWED_TYPES.
        """
        for i, funcs in enumerate(self._load_json(self.function_path)):
            try:
                func = Function(
                    name=funcs["name"],
                    description=funcs["description"],
                    parameters=funcs["parameters"],
                    nb_parameters=len(funcs["parameters"].keys()),
                    returns=funcs["returns"]
                )
            except (KeyError, TypeError, AttributeError, ValidationError):
                raise errors.InvalidFunctionDefinition(
                    f'{self.function_path}: entry {i} needs "name", '
                    '"description", "parameters" and "returns"'
                )
            if func.name in self.all_functions:
                raise errors.InvalidFunctionDefinition(
                    f'<{func.name}> is defined twice'
                )
            for p_name, p_type in func.parameters.items():
                if (not isinstance(p_type, dict)
                        or p_type.get("type") not in ALLOWED_TYPES):
                    raise errors.InvalidParameterValue(
                        f'Invalid type for parameter <{p_name}> '
                        f'of <{func.name}>'
                    )
            self.all_functions[func.name] = func
        if not self.all_functions:
            raise errors.InvalidFunctionDefinition(
                f"{self.function_path}: no function defined"
            )
