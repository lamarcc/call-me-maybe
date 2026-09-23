from __future__ import annotations
from llm_sdk import Small_LLM_Model
from enum import Enum, auto
import numpy as np
import errors
import torch


agent = Small_LLM_Model()


class Generator():
    path = agent.get_path_to_vocab_file()

    def __init__(self, functions, prompts) -> None:
        self.functions = functions
        self.prompts = prompts
        self.all_function_name: list = [function.name for function in [func for func in self.functions.values()]]
        self.all_function_description: list = [function.description for function in [func for func in self.functions.values()]]
        self.all_function_parameters: list = [function.parameters for function in [func for func in self.functions.values()]]
        self.all_function_returns: list = [function.returns for function in [func for func in self.functions.values()]]

    def get_allowed_function(self):
        allowed_tokens = self.all_function_name
        allowed_ids = []
        for name in allowed_tokens:
            encoded = agent.encode(name).tolist()[0]
            allowed_ids.extend(encoded)
        return np.array(allowed_ids, dtype=np.int32)

    def get_allowed_param(self):
        allowed_tokens = self.all_function_parameters
        allowed_ids = []
        for name in allowed_tokens:
            encoded = agent.encode(str(name)).tolist()[0]
            allowed_ids.extend(encoded)
        return np.array(allowed_ids, dtype=np.int32)

    def mask_logits(self, allowed_tokens, logits, already_generated):
        logits = np.asarray(logits, dtype=np.float32)
        masked = np.full_like(logits, -np.inf, dtype=np.float32)
        masked[allowed_tokens] = logits[allowed_tokens]
        masked[already_generated] = -np.inf
        return masked

    def generate_function_name(self, prompt):
        context = (
            "<|im_start|>system\n"
            "You are an AI Assistant that will help by giving\n"
            "the correct function name from a\n"
            "given list of known functions\n"
            "We dont want any text or thinking explanation\n"
            "only the function name\n"
            "Here are the known function:\n"
            f"{self.functions}"
            "<|im_end|>"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant<|im_end|>\n"
            "<think>\n\n</think>\n\n"
        )
        result = []
        already_generated = []
        context_tokenized = agent.encode(context).tolist()[0]
        allowed_function = self.get_allowed_function()
        while True:
            logits = agent.get_logits_from_input_ids(context_tokenized)
            mask = self.mask_logits(allowed_function, logits, already_generated)
            context += agent.decode([int(mask.argmax())])
            token_generated = int(mask.argmax())
            already_generated.append(token_generated)
            context_tokenized.append(token_generated)
            result.append(token_generated)
            if agent.decode(result) in self.all_function_name:
                return (agent.decode(result))

    def is_valid_value_type(self, value_list: list) -> bool:
        allowed_value_type: list[str] = [
            "number", "integer", "float", "string", "bool", "array"
        ]
        for verif in value_list:
            if verif not in allowed_value_type:
                return False
        return True

    def extract_param_value(self, function, prompt):
        parameter_name = []
        value_type = []
        for p_name in function.parameters.keys():
            parameter_name.append(p_name)
            for _, name in function.parameters[p_name].items():
                value_type.append(name)
        if not self.is_valid_value_type(value_type):
            raise errors.InvalidParameterValue(f'Invalid value type for <{function.name}>')
        return parameter_name, value_type

    def get_value(self, function, prompt, parameters, values) -> None:
        context = (
            "<|im_start|>system\n"
            "You are a function-calling engine.\n"
            "Given a user request and a function, you must:\n"
            "1. Return every parameter defined in function's schema, "
            "using exactly the parameter names given in the schema.\n"
            "3. For each parameter, return a value that matches exactly the type "
            "declared in the schema for that parameter (number, string, boolean, etc.).\n\n"

            "Rules:\n"
            "- Do not explain your reasoning.\n"
            "- Do not calculate, execute, transform, or answer the user's request.\n"
            "- Do not invent values that are not supported by the user's request.\n"
            "- Use exactly the parameter names as given in the function's schema.\n"
            "- Return every parameter defined in the schema, with no missing entries.\n"
            "- Return no extra parameters that are not defined in the schema.\n"
            "- Write each parameter as: parameter_name: \"parameter_value\"\n"
            "- Always wrap the value in double quotes, no matter its type.\n"
            "- Write one parameter per line.\n"
            "- Do not wrap the output in JSON, braces, or brackets.\n"
            "- After writing all the parameters, output the end-of-sequence token to stop generation.\n"
            "- Do not add any text before or after the parameter lines.\n\n"

            "The function to give the parameters from:\n"
            f"{function}\n\n"

            "Required output shape (one line per parameter):\n"
            '"parameter_name": "parameter_value"\n'
            '"parameter_name": "parameter_value"\n\n'

            "Example for string parameter:\n"
            "User prompt: Reverse the string 'hello'\n"
            "Output:\n"
            '"s": "hello"\n\n'

            "Example for number parameters:\n"
            "User prompt: Add 3 and 5\n"
            "Output:\n"
            '"a": "3.0"\n'
            '"b": "5.0"\n'
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
            logit = agent.get_logits_from_input_ids(context_tokenized)
            r = int(logit.index(max(logit)))
            context += agent.decode(r)
            result.append(r)
            context_tokenized.append(r)
            print(agent.decode(r))
