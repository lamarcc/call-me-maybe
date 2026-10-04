from __future__ import annotations
from src.masking import Mask
from src.state import GenerateJSON, State
from typing import Any
import numpy as np
import src.errors as errors
from src.errors import Colors
import json


class Generator():
    """
    Generate function calls with constrained decoding using token masking.
    """

    def __init__(self, agent: Any, functions: dict) -> None:
        """Initialize the Generator with an LLM agent and function definitions.

        Args:
            agent: LLM model instance with encode(), decode() and
                   get_logits_from_input_ids() methods.
            functions: Dictionary mapping function names to Function objects
                       with name, description, parameters, and returns
                       attributes.
        """
        self.agent = agent
        self.mask = Mask(agent)
        self.generate = GenerateJSON()
        self.functions = functions
        self.all_function_name: list = [
            function.name for function in self.functions.values()
        ]
        self.encoded_names: list[list[int]] = [
            agent.encode(name).tolist()[0] for name in self.all_function_name
        ]
        self.encoded_names.append(self._encode_string("None"))
        self.end_token: int = agent.encode("<|im_end|>").tolist()[0][0]

    def _next_token(
        self,
        context: np.ndarray,
        allowed: np.ndarray | None
    ) -> int:
        """Generate the next token from the model,
        restricted to allowed tokens.

        Args:
            context: Array of token IDs representing the prompt context.
            allowed: Array of allowed token IDs to sample from.

        Returns:
            The selected token ID with the highest
            probability among allowed tokens.
        """
        if allowed is not None and len(allowed) == 1:
            return int(allowed[0])
        logits = self.agent.get_logits_from_input_ids(context)
        return int(self.mask.mask_logits(allowed, logits).argmax())

    def _encode_string(self, string: str) -> Any:
        """Encode a string into token IDs.

        Args:
            string: Text to encode.

        Returns:
            List of token IDs.
        """
        return self.agent.encode(string).tolist()[0]

    def _decode_string(self, tokens: list) -> Any:
        """Decode token IDs back into a string.

        Args:
            tokens: List of token IDs to decode.

        Returns:
            The decoded text string.
        """
        return self.agent.decode(tokens)

    def _get_allowed_function(self, already_generated: list) -> Any:
        """Get allowed tokens for the next position in
        function name generation.

        Args:
            already_generated: List of token IDs already generated for the
                             function name.

        Returns:
            NumPy array of allowed token IDs that form valid function names
            or the end token.
        """
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
        """Generate a function name from a prompt using constrained decoding.

        Uses the LLM to select the most appropriate function from the available
        functions, constrained to only generate valid function name tokens.

        Args:
            prompt: User request text to determine the function.

        Returns:
            List of token IDs representing the selected function name.

        Raises:
            FunctionNotFound: If no known function can be
                              generated from the prompt.
        """
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
        """Extract parameter names and types from a function definition.

        Args:
            function: Function object with parameters attribute.
            prompt: User prompt (unused but kept for API consistency).

        Returns:
            Dictionary mapping parameter names to their types.
        """
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
        """Validate and convert a generated value to its correct type.

        Args:
            value: String representation of the generated value.
            parameter_name: Name of the parameter being validated.
            parameter_type: Expected type (integer, float, number,
                            boolean, string).

        Returns:
            The value converted to the correct Python type.

        Raises:
            InvalidGeneratedValue: If the value cannot be converted to the
                                  specified type.
        """
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
        """Generate a value for a single function parameter
        using constrained decoding.

        Args:
            function: Function object being called.
            prompt: User request text providing context.
            parameter_name: Name of the parameter to generate a value for.
            parameter_type: Expected type of the parameter value.
            already_extracted: Dictionary of previously extracted
            parameter values.

        Returns:
            The generated parameter value with correct type.

        Raises:
            InvalidGeneratedValue: If the value cannot be generated within the
                                  token limit or has an invalid type.
        """
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

    def build(self, prompt: str, all_functions: dict) -> Any:
        """Generate a complete JSON function call from a prompt.

        Orchestrates the entire generation process: selects the appropriate
        function, generates values for all its parameters, and builds a valid
        JSON structure containing the function call.

        Args:
            prompt: User request text describing the function call to generate.
            all_functions: List of available function definitions.

        Returns:
            Dictionary containing the generated function call with keys:
            - prompt: The input prompt
            - name: The selected function name
            - parameters: Dictionary of parameter names to generated values
        """
        context = (
            "<|im_start|>system\n"
            "Build a json file\n"
            "<|im_end|>\n"
        )
        result = []
        self.generate.set_start()
        function = self._decode_string(self._generate_function_name(prompt))
        print(f"   Building prompt: {prompt}")
        if function == "None":
            print(Colors.FAIL + "   FAILED" + Colors.ENDC, end=" - ")
            raise errors.GenerationError("No known function for")
        function = self.functions[function]
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
        while self.generate.get_state() != State.FINISH:
            state = self.generate.get_state()
            state_value = self.generate.get_value()
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
                self.generate._actual_state = State.COMMA_AFTER_PARAMETER_VALUE
            else:
                self.generate.next_state(self.generate.get_state())
        print(Colors.OKGREEN + "   SUCCESS" + Colors.ENDC)
        return json.loads(self._decode_string(result))
