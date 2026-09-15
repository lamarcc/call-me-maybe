from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Optional
import json
import errors


class Prompt(BaseModel):
    prompt: str = Field(min_length=1)


class Function(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, Any]
    returns: dict[str, str]


class Parse():
    def __init__(self, f_path: Optional[str], p_path: Optional[str]) -> None:
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

    def prompt(self) -> None:
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as file:
                f_json: str = json.load(file)
                for prompts in f_json:
                    prompt = Prompt(prompt=prompts["prompt"])
                    self.all_prompts.append(prompt)
        except json.JSONDecodeError:
            raise errors.InvalidJSON("Invalid JSON file")
        except PermissionError:
            raise errors.NoPermissionErr("Permission denied")
        except FileNotFoundError:
            raise errors.FileNotFoundErr("The file does not exist")

    def function(self) -> None:
        try:
            with open(self.function_path, "r", encoding="utf-8") as file:
                f_json: str = json.load(file)
                for funcs in f_json:
                    func = Function(
                        name=funcs["name"],
                        description=funcs["description"],
                        parameters=funcs["parameters"],
                        returns=funcs["returns"]
                    )
                    self.all_functions[func.name] = func
        except json.JSONDecodeError:
            print("Not a valid json")
        except PermissionError:
            print("No permission")
        except FileNotFoundError:
            print("No file found")
