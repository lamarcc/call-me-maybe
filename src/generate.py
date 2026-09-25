from __future__ import annotations
from llm_sdk import Small_LLM_Model
from masking import Mask
import numpy as np
import errors
import torch


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
        name = ""
        for function in self.functions.values():
            name += f"- {function.name}: {function.description}\n"
        context = (
            "<|im_start|>system\n"
            "You are an AI Assistant that will help by giving\n"
            "the correct function name from a\n"
            "given list of known functions\n"
            "We dont want any text or thinking explanation\n"
            "only the function name\n"
            "Here are the known function:\n"
            f"{name}"
            "<|im_end|>"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant<|im_end|>\n"
            "<think>\n\n</think>\n\n"
        )
        result = []
        context_tokenized = agent.encode(context).tolist()[0]
        while True:
            allowed = self.get_allowed_function(result)
            logits = agent.get_logits_from_input_ids(context_tokenized)
            mask = self.mask.mask_logits(allowed, logits)
            token_generated = int(mask.argmax())
            context_tokenized.append(token_generated)
            result.append(token_generated)
            if agent.decode(result) in self.all_function_name:
                return (agent.decode(result))

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

    def get_value(self, function, prompt, parameter, values) -> None:
        context = (
            "<|im_start|>system\n"
            "You are a parameter-value extraction engine.\n"
            "Your ONLY task is to extract the RAW "
            "value from the user's request.\n"
            "DO NOT execute, calculate, or transform the value.\n"
            "DO NOT return the result of the function.\n"
            "Just extract the exact value mentioned in the request.\n\n"
            f"Function: '{function.name}' - {function.description}\n"
            f"Target parameter: '{parameter}'\n\n"
            "CRITICAL RULES:\n"
            "- Return ONLY the raw value as it appears in the request.\n"
            "- Do NOT include the parameter name.\n"
            "- Do NOT write 'parameter: value' or 'parameter=value'.\n"
            "- Just the value, nothing else.\n\n"
            "Examples:\n"
            "User: 'Reverse the string hello'\n"
            "Parameter: 'text'\n"
            "Correct output: hello\n"
            "Wrong output: olleh (this is the result, not the raw value)\n\n"
            "User: 'Sum of 265 and 345'\n"
            "Parameter: 'a'\n"
            "Correct output: 265\n"
            "Wrong output: 610 (this is the result, not the raw value)\n\n"
            "<|im_end|>\n"

            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"

            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        result = []
        context_tokenized = agent.encode(context).tolist()[0]
        while True:
            current = agent.decode(result)
            allowed = self.mask.get_allowed_type(values, current)
            full = context_tokenized + result
            logits = agent.get_logits_from_input_ids(full)
            mask = self.mask.mask_logits(allowed, logits)
            r = int(mask.argmax())
            result.append(r)
            print(current)
            if "}" in agent.decode(result):
                return agent.decode(result)
