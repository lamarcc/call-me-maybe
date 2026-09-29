from __future__ import annotations
from llm_sdk import Small_LLM_Model
from masking import Mask
from output import GenerateJSON, State
import numpy as np
import errors
import json


agent = Small_LLM_Model()


class Generator():
    path = agent.get_path_to_vocab_file()

    def __init__(self, functions, prompts) -> None:
        self.mask = Mask(agent)
        self.functions = functions
        self.prompts = prompts
        self.all_function_name: list = [
            function.name for function in self.functions.values()
        ]

    def get_allowed_function(self, already_generated):
        allowed_tokens = self.all_function_name
        allowed = []
        for name in allowed_tokens:
            encoded = agent.encode(name).tolist()[0]
            if already_generated == encoded[:len(already_generated)]:
                if len(already_generated) < len(encoded):
                    allowed.append(encoded[len(already_generated)])
        return np.array(allowed, dtype=np.int32)

    def generate_function_name(self, prompt):
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
            f"{prompt.prompt}<|im_end|>\n"
            "<|im_start|>assistant<|im_end|>\n"
            "<think>\n\n</think>\n\n"
        )
        result: list = []
        context_tokenized = agent.encode(context).tolist()[0]
        while True:
            allowed = self.get_allowed_function(result)
            logits = agent.get_logits_from_input_ids(context_tokenized)
            mask = self.mask.mask_logits(allowed, logits)
            token_generated = int(mask.argmax())
            context_tokenized.append(token_generated)
            result.append(token_generated)
            if agent.decode(result) in self.all_function_name:
                return (result)

    def is_valid_value_type(self, value_list: list) -> bool:
        allowed_value_type: list[str] = [
            "number", "integer", "float", "string", "boolean"
        ]
        for verif in value_list:
            if verif not in allowed_value_type:
                return False
        return True

    def extract_param_value(self, function, prompt):
        parameters = {}
        value_type = []
        for p_name, p_type in function.parameters.items():
            parameters[p_name] = p_type["type"]
        if not self.is_valid_value_type(value_type):
            raise errors.InvalidParameterValue(
                f'Invalid value type for <{function.name}>'
            )
        return parameters

    def check_value(self, state, prompt):
        if state is State.FUNCTION_NAME_VALUE:
            return self.generate_function_name(prompt)

    def build(self, prompt, all_functions):
        context = (
            "<|im_start|>system\n"
            "Build a json file\n"
            "<|im_end|>\n"
        )
        generate = GenerateJSON()
        result = []
        context_tokenized = agent.encode(context).tolist()[0]
        while generate.get_state() != State.FINISH:
            state = generate.get_state()
            value = generate.get_value()
            if state is State.PROMPT_VALUE:
                p_token = agent.encode(prompt.prompt).tolist()[0]
                for token in p_token:
                    result.append(token)
            elif state is State.FUNCTION_NAME_VALUE:
                f_token = self.generate_function_name(prompt)
                param_left = all_functions[agent.decode(f_token)].nb_parameters
                for token in f_token:
                    result.append(token)
            else:
                allowed = self.mask._get_token(value)
                logits = agent.get_logits_from_input_ids(context_tokenized)
                mask = self.mask.mask_logits(allowed, logits)
                token = int(mask.argmax())
                result.append(token)
                context_tokenized.append(token)
            if state is State.PARAMETER_VALUE and param_left > 1:
                param_left -= 1
                generate._actual_state = State.COMMA_AFTER_PARAMETER_VALUE
            else:
                generate.next_state(generate.get_state())
            print(agent.decode(result))

    def get_value(self, function, prompt, parameter_name, parameter_type) -> None:
        context = (
            "<|im_start|>system\n"
            f'Function: {function.name} {function.description}\n'
            f'User Prompt: {prompt.prompt}\n'
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Prompt: {prompt.prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
            'Arguments JSON: {"' + f'{parameter_name}": "'
        )
        result = []
        context_tokenized = agent.encode(context).tolist()[0]
        try:
            while True:
                current = agent.decode(result)
                allowed = self.mask.get_allowed_type(parameter_type, current)
                full = context_tokenized + result
                logits = agent.get_logits_from_input_ids(full)
                mask = self.mask.mask_logits(allowed, logits)
                r = int(mask.argmax())
                result.append(r)
                print(agent.decode(result))
                if '"' in agent.decode(result) or "Human" in agent.decode(result):
                    return agent.decode(result)
        except KeyboardInterrupt:
            exit(1)
