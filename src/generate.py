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
        allowed_tokens = self.all_function_description
        allowed_ids = []
        for name in allowed_tokens:
            encoded = agent.encode(name).tolist()[0]
            allowed_ids.extend(encoded)
        return np.array(allowed_ids, dtype=np.int32)

    def mask_logits(self, allowed, logits):
        logits = np.asarray(logits, dtype=np.float32)
        masked = np.full_like(logits, -np.inf, dtype=np.float32)
        masked[allowed] = logits[allowed]
        return masked

    def generate_function_name(self, prompt):
        context = (
            "<|im_start|>system\n"
            "You are an AI Assistant that will help by giving\n"
            "the correct function name and parameter from a\n"
            "given list of function known\n"
            "We dont want any text or thinking explanation\n"
            "only the function name and parameters"
            "Here are the known function:\n"
            f"{self.functions}"
            "<|im_end|>"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant<|im_end|>\n"
            "<think>\n\n</think>\n\n"
        )
        result = []
        token = agent.encode(context).tolist()[0]
        while True:
            logit = agent.get_logits_from_input_ids(token)
            allowed = self.get_allowed_function()
            mask = self.mask_logits(allowed, logit)
            self.context += agent.decode([int(mask.argmax())])
            result.append(int(mask.argmax()))
            token.append(int(mask.argmax()))
            if agent.decode(result) in self.all_function_name:
                print(agent.decode(result))
                break
