from __future__ import annotations
from llm_sdk import Small_LLM_Model
from masking import Mask
from output import GenerateJSON, State
import numpy as np
import errors


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
        self.encoded_names: list[list[int]] = [
            agent.encode(name).tolist()[0] for name in self.all_function_name
        ]
        self.end_token: int = agent.encode("<|im_end|>").tolist()[0][0]

    def get_allowed_function(self, already_generated):
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
            if len(allowed) == 0:
                break
            if len(allowed) == 1:
                token_generated = int(allowed[0])
            else:
                logits = agent.get_logits_from_input_ids(context_tokenized)
                mask = self.mask.mask_logits(allowed, logits)
                token_generated = int(mask.argmax())
            if token_generated == self.end_token:
                return result
            context_tokenized.append(token_generated)
            result.append(token_generated)
        raise errors.FunctionNotFound(
            'No known function matches '
            f'(generated: "{agent.decode(result)}")'
        )

    def extract_param_value(self, function, prompt):
        return {
            p_name: p_type["type"]
            for p_name, p_type in function.parameters.items()
        }

    def check_generated_value(self, value: str, parameter_name: str,
                              parameter_type: str) -> None:
        try:
            if parameter_type == "integer":
                int(value)
            elif parameter_type in ("float", "number"):
                float(value)
            elif parameter_type == "boolean" and value not in ("true", "false"):
                raise ValueError
        except ValueError:
            raise errors.InvalidGeneratedValue(
                f'"{value}" is not a valid {parameter_type} '
                f'for parameter <{parameter_name}>'
            )

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
        function = self.functions[agent.decode(self.generate_function_name(prompt))]
        context_tokenized = agent.encode(context).tolist()[0]
        while generate.get_state() != State.FINISH:
            state = generate.get_state()
            value = generate.get_value()
            if state is State.PROMPT_VALUE:
                p_token = agent.encode(prompt.prompt).tolist()[0]
                for token in p_token:
                    result.append(token)
            elif state is State.FUNCTION_NAME_VALUE:
                f_token = agent.encode(function.name).tolist()[0]
                param_left = function.nb_parameters
                parameter_name_list = [name for name in iter(self.extract_param_value(function, prompt))]
                for token in f_token:
                    result.append(token)
            elif state is State.PARAMETER_NAME:
                param_token = agent.encode(parameter_name_list[function.nb_parameters - param_left]).tolist()[0]
                for token in param_token:
                    result.append(token)
            else:
                allowed = self.mask._get_token(value)
                logits = agent.get_logits_from_input_ids(context_tokenized)
                mask = self.mask.mask_logits(allowed, logits)
                token = int(mask.argmax())
                result.append(token)
                context_tokenized.append(token)
            if state is State.QUOTE_AFTER_PARAM_VALUE and param_left > 1:
                param_left -= 1
                generate._actual_state = State.COMMA_AFTER_PARAMETER_VALUE
            else:
                generate.next_state(generate.get_state())
            print(agent.decode(result))

    def get_value(self, function, prompt, parameter_name, parameter_type, max_tokens: int = 50, already_extracted: dict | None = None) -> str:
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
            f'User Prompt: {prompt.prompt}\n'
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Prompt: {prompt.prompt}\n"
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
        context_tokenized = agent.encode(context).tolist()[0]
        try:
            for _ in range(max_tokens):
                current = agent.decode(result)
                allowed = self.mask.get_allowed_type(parameter_type, current)
                full = context_tokenized + result
                logits = agent.get_logits_from_input_ids(full)
                mask = self.mask.mask_logits(allowed, logits)
                r = int(mask.argmax())
                result.append(r)
                current_decoded = agent.decode(result)
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
            self.check_generated_value(value, parameter_name, parameter_type)
            return value
        except KeyboardInterrupt:
            exit(1)
