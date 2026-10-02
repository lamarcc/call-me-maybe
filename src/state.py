from __future__ import annotations
from enum import Enum, auto


class Vocab():
    OPEN_BRACE = '{'
    CLOSE_BRACE = '}'
    COLON = ':'
    COMMA = ','
    QUOTE = '"'


class State(Enum):
    OPEN_START = auto()
    OPEN_PARAMETER = auto()
    PROMPT_KEY = auto()
    FUNCTION_NAME_KEY = auto()
    PARAMETER_KEY = auto()
    COLON_AFTER_PROMPT_KEY = auto()
    COLON_AFTER_FUNCTION_NAME_KEY = auto()
    COLON_AFTER_PARAMETER_KEY = auto()
    COLON_AFTER_PARAMETER_NAME = auto()
    COMMA_AFTER_PROMPT_VALUE = auto()
    COMMA_AFTER_FUNCTION_NAME_VALUE = auto()
    COMMA_AFTER_PARAMETER_VALUE = auto()
    QUOTE_BEFORE_PKEY = auto()
    QUOTE_AFTER_PKEY = auto()
    QUOTE_BEFORE_PVALUE = auto()
    QUOTE_AFTER_PVALUE = auto()
    QUOTE_BEFORE_FKEY = auto()
    QUOTE_AFTER_FKEY = auto()
    QUOTE_BEFORE_FVALUE = auto()
    QUOTE_AFTER_FVALUE = auto()
    QUOTE_BEFORE_PAKEY = auto()
    QUOTE_AFTER_PAKEY = auto()
    QUOTE_BEFORE_PARAM_NAME = auto()
    QUOTE_AFTER_PARAM_NAME = auto()
    PROMPT_VALUE = auto()
    FUNCTION_NAME_VALUE = auto()
    PARAMETER_NAME = auto()
    PARAMETER_VALUE = auto()
    CLOSE_END = auto()
    CLOSE_PARAMETER = auto()
    FINISH = auto()


class GenerateJSON():
    def __init__(self) -> None:
        self._actual_state: State = State.OPEN_START
        self.step: dict[State, State] = {
            State.OPEN_START: State.QUOTE_BEFORE_PKEY,
            State.QUOTE_BEFORE_PKEY: State.PROMPT_KEY,
            State.PROMPT_KEY: State.QUOTE_AFTER_PKEY,
            State.QUOTE_AFTER_PKEY: State.COLON_AFTER_PROMPT_KEY,
            State.COLON_AFTER_PROMPT_KEY: State.QUOTE_BEFORE_PVALUE,
            State.QUOTE_BEFORE_PVALUE: State.PROMPT_VALUE,
            State.PROMPT_VALUE: State.QUOTE_AFTER_PVALUE,
            State.QUOTE_AFTER_PVALUE: State.COMMA_AFTER_PROMPT_VALUE,
            State.COMMA_AFTER_PROMPT_VALUE: State.QUOTE_BEFORE_FKEY,
            State.QUOTE_BEFORE_FKEY: State.FUNCTION_NAME_KEY,
            State.FUNCTION_NAME_KEY: State.QUOTE_AFTER_FKEY,
            State.QUOTE_AFTER_FKEY: State.COLON_AFTER_FUNCTION_NAME_KEY,
            State.COLON_AFTER_FUNCTION_NAME_KEY: State.QUOTE_BEFORE_FVALUE,
            State.QUOTE_BEFORE_FVALUE: State.FUNCTION_NAME_VALUE,
            State.FUNCTION_NAME_VALUE: State.QUOTE_AFTER_FVALUE,
            State.QUOTE_AFTER_FVALUE: State.COMMA_AFTER_FUNCTION_NAME_VALUE,
            State.COMMA_AFTER_FUNCTION_NAME_VALUE: State.QUOTE_BEFORE_PAKEY,
            State.QUOTE_BEFORE_PAKEY: State.PARAMETER_KEY,
            State.PARAMETER_KEY: State.QUOTE_AFTER_PAKEY,
            State.QUOTE_AFTER_PAKEY: State.COLON_AFTER_PARAMETER_KEY,
            State.COLON_AFTER_PARAMETER_KEY: State.OPEN_PARAMETER,
            State.OPEN_PARAMETER: State.QUOTE_BEFORE_PARAM_NAME,
            State.QUOTE_BEFORE_PARAM_NAME: State.PARAMETER_NAME,
            State.PARAMETER_NAME: State.QUOTE_AFTER_PARAM_NAME,
            State.QUOTE_AFTER_PARAM_NAME: State.COLON_AFTER_PARAMETER_NAME,
            State.COLON_AFTER_PARAMETER_NAME: State.PARAMETER_VALUE,
            State.PARAMETER_VALUE: State.CLOSE_PARAMETER,
            State.CLOSE_PARAMETER: State.CLOSE_END,
            State.CLOSE_END: State.FINISH,
            State.FINISH: None,
            State.COMMA_AFTER_PARAMETER_VALUE: State.QUOTE_BEFORE_PARAM_NAME
        }
        self.value: dict[State, str] = {
            State.OPEN_START: Vocab.OPEN_BRACE,
            State.OPEN_PARAMETER: Vocab.OPEN_BRACE,
            State.PROMPT_KEY: "prompt",
            State.FUNCTION_NAME_KEY: "name",
            State.PARAMETER_KEY: "parameters",
            State.COLON_AFTER_PROMPT_KEY: Vocab.COLON,
            State.COLON_AFTER_FUNCTION_NAME_KEY: Vocab.COLON,
            State.COLON_AFTER_PARAMETER_KEY: Vocab.COLON,
            State.COLON_AFTER_PARAMETER_NAME: Vocab.COLON,
            State.COMMA_AFTER_PROMPT_VALUE: Vocab.COMMA,
            State.COMMA_AFTER_FUNCTION_NAME_VALUE: Vocab.COMMA,
            State.COMMA_AFTER_PARAMETER_VALUE: Vocab.COMMA,
            State.QUOTE_BEFORE_PKEY: Vocab.QUOTE,
            State.QUOTE_AFTER_PKEY: Vocab.QUOTE,
            State.QUOTE_BEFORE_PVALUE:  Vocab.QUOTE,
            State.QUOTE_AFTER_PVALUE: Vocab.QUOTE,
            State.QUOTE_BEFORE_FKEY: Vocab.QUOTE,
            State.QUOTE_AFTER_FKEY: Vocab.QUOTE,
            State.QUOTE_BEFORE_FVALUE: Vocab.QUOTE,
            State.QUOTE_AFTER_FVALUE: Vocab.QUOTE,
            State.QUOTE_BEFORE_PAKEY: Vocab.QUOTE,
            State.QUOTE_AFTER_PAKEY: Vocab.QUOTE,
            State.QUOTE_BEFORE_PARAM_NAME: Vocab.QUOTE,
            State.QUOTE_AFTER_PARAM_NAME: Vocab.QUOTE,
            State.PROMPT_VALUE: "None",
            State.FUNCTION_NAME_VALUE: "None",
            State.PARAMETER_NAME: "None",
            State.PARAMETER_VALUE: "None",
            State.CLOSE_END: Vocab.CLOSE_BRACE,
            State.CLOSE_PARAMETER: Vocab.CLOSE_BRACE,
            State.FINISH: ""
        }

    def get_state(self) -> State:
        return self._actual_state

    def get_value(self) -> str:
        return self.value[self._actual_state]

    def next_state(self, actual: State) -> State:
        if actual is None:
            return
        self._actual_state = self.step[actual]
        return self.step[actual]
