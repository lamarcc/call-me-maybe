from __future__ import annotations
from llm_sdk import Small_LLM_Model
from abc import ABC
import errors


agent = Small_LLM_Model()


class Vocab():
    path = agent.get_path_to_vocab_file()

    def __init__(self) -> None:
        self.mask: dict = {}


class Mask(ABC):
    def __init__(self):
        pass

    def validate(self, token: str):
        pass

    def get_valid_token(self):
        pass


class BoolMask(Mask):
    pass


class IntegerMask(Mask):
    pass


class StringMask(Mask):
    pass
