from __future__ import annotations
from llm_sdk import Small_LLM_Model
from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np
import errors


agent = Small_LLM_Model()


class Vocab():
    path = agent.get_path_to_vocab_file()
    bool_mask: np.ndarray
    int_mask: np.ndarray
    str_mask: np.ndarray

    def set_score(self, mask: np.ndarray) -> np.ndarray:



class Mask(ABC):
    @abstractmethod
    def validate(self, token: int) -> bool:
        pass

    @abstractmethod
    def get_valid_token(self) -> list:
        pass


class BoolMask(Mask):
    def validate(self, token: int) -> bool:
        verif = agent.decode(token)
        return (verif.lower() in ["true", "false", "0", "1"])

    def get_valid_token(self) -> np.ndarray:
        return np.array(["0", "1", "true", "false"])


class IntegerMask(Mask):
    def validate(self, token: int) -> bool:
        verif = agent.decode(token)
        return (verif in "0123456789")

    def get_valid_token(self) -> np.ndarray:
        return ["0123456789"]


class StringMask(Mask):
    def validate(self, token: int) -> bool:
        return True


class State(Enum):
    START = auto()
