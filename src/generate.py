from __future__ import annotations
from src.masking import Mask
from src.state import GenerateJSON, State
import numpy as np
import src.errors as errors
import json
from typing import Any


class Generator():
    def __init__(self, agent: Any, functions: dict) -> None:
        self.agent = agent
        self.mask = Mask(agent)
        self.functions = functions
        self.all_function_name: list = [
            function.name for function in self.functions.values()
        ]
        self.encoded_names: list[list[int]] = [
            agent.encode(name).tolist()[0] for name in self.all_function_name
        ]
        self.end_token: int = agent.encode("<|im_end|>").tolist()[0][0]

    def _next_token(self, context: np.ndarray, allowed: np.ndarray) -> int:
        if allowed is not None and len(allowed) == 1:
            return int(allowed[0])
        logits = self.agent.get_logits_from_input_ids(context)
        return int(self.mask.mask_logits(allowed, logits).argmax())

    def _encode_string(self, string: str) -> Any:
        return self.agent.encode(string).tolist()[0]

    def _decode_string(self, tokens: list) -> Any:
        return self.agent.decode(tokens)

    def _get_allowed_function(self, already_generated: list) -> Any:
        allowed = set()
        size = len(already_generated)
        for encoded in self.encoded_names:
            if encoded[:size] != already_generated:
                continue
            if size < len(encoded):
                allowed.add(encoded[size])
            else:
                allowed.add(self.end_token)
        return np.array(list(allowed), dtype=np.int32)

    def _generate_function_name(self, prompt: str) -> list:
        names: str = ""
        for function in self.functions.values():
            names += f"- {function.name}: {function.description}\n"
        context = (
            "<|im_start|>system\n"
            "You are an AI Assistant that will help by giving\n"
            "the correct function name from a\n"
            "given list of known functions\n"
            "We dont want any text or thinking explanation\n"
            "only the function name\n"
            "Here are the known function:\n"
            f"{names}"
            "<|im_end|>"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant<|im_end|>\n"
            "<think>\n\n</think>\n\n"
        )
        result: list = []
        context_tokenized = self._encode_string(context)
        while True:
            allowed = self._get_allowed_function(result)
            if len(allowed) == 0:
                break
            token_generated = self._next_token(context_tokenized, allowed)
            if token_generated == self.end_token:
                return result
            context_tokenized.append(token_generated)
            result.append(token_generated)
        raise errors.FunctionNotFound(
            'No known function matches '
            f'(generated: "{self._decode_string(result)}")'
        )

    def _extract_param_value(self, function: Any, prompt: str) -> Any:
        return {
            p_name: p_type["type"]
            for p_name, p_type in function.parameters.items()
        }

    def _check_generated_value(
        self,
        value: str,
        parameter_name: str,
        parameter_type: str
    ) -> str | int | float:
        try:
            if parameter_type == "integer":
                return int(value)
            elif parameter_type in ("float", "number"):
                return float(value)
            elif parameter_type == "boolean" and value not in (
                    "true",
                    "false"
            ):
                raise ValueError
            return value
        except ValueError:
            raise errors.InvalidGeneratedValue(
                f'"{value}" is not a valid {parameter_type} '
                f'for parameter <{parameter_name}>'
            )

    def _get_value(
            self,
            function: Any,
            prompt: str,
            parameter_name: str,
            parameter_type: str,
            already_extracted: dict | None = None
    ) -> str | int | float:
        args_prefix = ""
        if already_extracted:
            args_prefix = ", ".join(
                f'"{k}": "{v}"' for k, v in already_extracted.items()
            )
            if args_prefix:
                args_prefix += ", "
        context = (
            "<|im_start|>system\n"
            f'Function: {function.name} {function.description}\n'
            f'User Prompt: {prompt}\n'
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Prompt: {prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
            f'Arguments JSON: {{{args_prefix}"{parameter_name}":'
        )
        numeric = parameter_type in ("integer", "float", "number")
        stop_chars = '",}\n' if numeric else '"'
        if not numeric:
            context += ' "'
        result: list = []
        context_tokenized = self._encode_string(context)
        max_tokens = 50
        for _ in range(max_tokens):
            current = self._decode_string(result)
            allowed = self.mask._get_allowed_type(parameter_type, current)
            result.append(
                self._next_token(context_tokenized + result, allowed)
            )
            current_decoded = self._decode_string(result)
            if any(c in current_decoded for c in stop_chars):
                break
        else:
            raise errors.InvalidGeneratedValue(
                f'Parameter <{parameter_name}> not closed after '
                f'{max_tokens} tokens'
            )
        for c in stop_chars:
            current_decoded = current_decoded.split(c)[0]
        value = current_decoded.strip('",} \n\r\t')
        return self._check_generated_value(
            value,
            parameter_name,
            parameter_type
        )

    def build(self, prompt: str, all_functions: list) -> Any:
        context = (
            "<|im_start|>system\n"
            "Build a json file\n"
            "<|im_end|>\n"
        )
        generate = GenerateJSON()
        result = []
        function = self.functions[self.agent.decode(
            self._generate_function_name(prompt)
        )]
        extracted: dict = {}
        param_list: list = []
        type_list: list = []
        value_list: list = []
        params = self._extract_param_value(function, prompt)
        for p_name, p_type in params.items():
            value = self._get_value(
                function,
                prompt,
                p_name,
                p_type,
                already_extracted=extracted
            )
            extracted[p_name] = value
            param_list.append(p_name)
            type_list.append(p_type)
            value_list.append(str(value))
        context_tokenized = self._encode_string(context)
        while generate.get_state() != State.FINISH:
            state = generate.get_state()
            state_value = generate.get_value()
            if state is State.PROMPT_VALUE:
                p_token = self._encode_string(json.dumps(prompt)[1:-1])
                for token in p_token:
                    result.append(token)
            elif state is State.FUNCTION_NAME_VALUE:
                f_token = self._encode_string(function.name)
                param_left = function.nb_parameters
                for token in f_token:
                    result.append(token)
            elif state is State.PARAMETER_NAME:
                param_token = self._encode_string(param_list[0])
                param_list.pop(0)
                for token in param_token:
                    result.append(token)
            elif state is State.PARAMETER_VALUE:
                value_token = self._encode_string(value_list[0])
                value_list.pop(0)
                if type_list[0] == "string" or type_list[0] == "boolean":
                    value_token.insert(0, 1)
                    value_token.append(1)
                type_list.pop(0)
                for token in value_token:
                    result.append(token)
            else:
                allowed = self.mask.get_token(state_value)
                token = self._next_token(context_tokenized, allowed)
                result.append(token)
                context_tokenized.append(token)
            if state is State.PARAMETER_VALUE and param_left > 1:
                param_left -= 1
                generate._actual_state = State.COMMA_AFTER_PARAMETER_VALUE
            else:
                generate.next_state(generate.get_state())
        return json.loads(self._decode_string(result))
