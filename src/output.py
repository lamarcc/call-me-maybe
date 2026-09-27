from llm_sdk import Small_LLM_Model
from enum import Enum, auto
import torch

agent = Small_LLM_Model()


class JsonState(Enum):
    OPEN = auto()
    FUNCTION_NAME_KEY = auto()
    COLON = auto()
    FUNCTION_NAME = auto()
    COMMA = auto()
    PARAMETER_KEY = auto()
    COLON = auto()
    OPEN = auto()
    PARAMETER_VALUE = auto()
    CLOSE = auto()
    CLOSE = auto()
