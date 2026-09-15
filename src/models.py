from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any
import json


class Function(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict
    returns: Any


class Parser():
    def __init__(self) -> None:
        pass

    def function(self) -> None:
        with open("data/input/functions_definition.json", "r", encoding="utf-8") as file:
            print(json.load(file))
