from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional
import json


class Function(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict
    returns: Any

    @model_validator(mode='after')
    def checker(self):
        pass


class Parser():
    def __init__(self, input: Optional) -> None:
        if input:
            self.json_path: str = input
        else:
            self.json_path: str = "data/input/functions_definition.json"
        self.all_functions: dict = {}

    def function(self) -> None:
        try:
            with open(self.json_path, "r", encoding="utf-8") as file:
                f_json: str = json.load(file)
                for funcs in f_json:
                    func = Function(name=funcs["name"], description=funcs["description"], parameters=funcs["parameters"], returns=funcs["returns"])
                    self.all_functions[func.name] = func
        except json.JSONDecodeError:
            print("Not a valid json")
        except PermissionError:
            print("No permission")
        except FileNotFoundError:
            print("No file found")
