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

    def generate_param(self, function, prompt):
        p_name = []
        typeu = []
        for parameter_name in function.parameters.keys():
            p_name.append(parameter_name)
            for _, name in function.parameters[parameter_name].items():
                typeu.append(name)
        print(p_name)
        print(typeu)
        context = (
            "<|im_start|>system\n"
            "You are a parameter-typing engine for function calls.\n"
            "Your only task is to determine, for each parameter defined in the schema below, "
            "the correct type and the correct value, strictly based on the user's request.\n\n"

            "Rules:\n"
            "- Return ONLY the parameter values. Do not return the function name.\n"
            "- Do not explain your reasoning.\n"
            "- Do not calculate, execute, transform, or answer the user's request.\n"
            "- Do not invent values that are not supported by the user's request.\n"
            "- For each parameter in the schema, return its type and its value, nothing else.\n"
            "- Do not return parameter names, only their type and value.\n"
            "- Return values for every parameter defined in the schema, with no missing entries.\n"
            "- Return no extra entries that are not defined in the schema.\n"
            "- Each value must match exactly the type declared in the schema for that parameter.\n"
            "- If a parameter type is number, output a JSON number in float format, without quotes.\n"
            "- If a parameter type is string, output a JSON string, with double quotes.\n"
            "- If a parameter type is boolean, output a JSON boolean (true or false), without quotes.\n"
            "- Output must be valid JSON and must contain only the JSON object, nothing before or after.\n\n"

            f"Function name:\n{function.name}\n\n"
            f"Function description:\n{function.description}\n\n"
            f"Parameter name:\n{p_name}\n\n"
            f"Parameter value type:\n{typeu}\n\n"

            "Required output shape:\n"
            '{"parameter_name": parameter_value}\n'
            "<|im_end|>\n"

            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"

            "<|im_start|>assistant<|im_end|>\n"
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
            if "}" in agent.decode(r) or agent.decode(r) is None:
                return (agent.decode(result))
