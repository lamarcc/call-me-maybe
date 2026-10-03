from typing import Any
import numpy as np


class Mask():
    """Restrict token generation based on JSON structure and parameter types."""

    def __init__(self, agent: Any) -> None:
        """Initialize the mask with encoding agent and closing tokens.

        Args:
            agent: LLM model instance with encode() method for tokenization.
        """
        self.agent = agent
        closers = ['"', '",', '"}', '"\n', '",\n', '}\n', '}', ',']
        tokens: set = set()
        for c in closers:
            tokens.update(self.get_token(c))
        self.closing_tokens: np.ndarray = np.array(
            list(tokens), dtype=np.int32
        )

    def get_token(self, text: str) -> np.ndarray:
        """Encode text into token IDs.

        Args:
            text: Text string to encode.

        Returns:
            NumPy array of token IDs.
        """
        token = self.agent.encode(text).tolist()
        if isinstance(token[0], list):
            token = token[0]
        return np.array(token, dtype=np.int32)

    def _get_int(self, value: str) -> np.ndarray:
        """Get allowed tokens for generating an integer value.

        Args:
            value: Current value string being generated.

        Returns:
            NumPy array of allowed token IDs for integer generation.
        """
        allowed = list('0123456789')
        number = value.strip()
        if value == "":
            allowed.append(" ")
        if number == "":
            allowed.extend(["-", " -"])
        allowed_token: set = set()
        for c in allowed:
            allowed_token.update(self.get_token(c))
        if number not in ["", "-"]:
            allowed_token.update(self.closing_tokens.tolist())
        return np.array(list(allowed_token), dtype=np.int32)

    def _get_float(self, value: str) -> np.ndarray:
        """Get allowed tokens for generating a floating-point value.

        Args:
            value: Current value string being generated.

        Returns:
            NumPy array of allowed token IDs for float generation.
        """
        allowed = list('0123456789')
        number = value.strip()
        if "." not in number and number not in ["", "-"]:
            allowed.append(".")
        if value == "":
            allowed.append(" ")
        if number == "":
            allowed.extend(["-", " -"])
        allowed_token: set = set()
        for c in allowed:
            allowed_token.update(self.get_token(c))
        if number not in ["", "-"] and not number.endswith("."):
            allowed_token.update(self.closing_tokens.tolist())
        return np.array(list(allowed_token), dtype=np.int32)

    def _get_boolean(self, value: str = "") -> np.ndarray:
        """Get allowed tokens for generating a boolean value.

        Args:
            value: Current value string being generated (unused).

        Returns:
            NumPy array of allowed token IDs for boolean values (true/false).
        """
        allowed: set = set()
        for word in ["true", "false"]:
            allowed.update(self.get_token(word))
        allowed.update(self.closing_tokens.tolist())
        return np.array(list(allowed), dtype=np.int32)

    def _get_allowed_type(
        self, value_type: str, value: str
    ) -> np.ndarray | None:
        """Get allowed tokens based on the parameter type.

        Args:
            value_type: Type of value (integer, float, number, boolean, string).
            value: Current value string being generated.

        Returns:
            NumPy array of allowed token IDs, or None for unrestricted strings.

        Raises:
            ValueError: If value_type is not a recognized type.
        """
        if value_type == "integer":
            return self._get_int(value)
        if value_type == "float" or value_type == "number":
            return self._get_float(value)
        if value_type == "boolean":
            return self._get_boolean(value)
        if value_type == "string":
            return None
        raise ValueError(f"Unknown type: {value_type}")

    def mask_logits(
        self, allowed: np.ndarray, logits: np.ndarray
    ) -> np.ndarray:
        """Apply mask to logits, restricting to allowed tokens.

        Sets logits of disallowed tokens to negative infinity so they cannot
        be selected, while keeping allowed tokens' logits unchanged.

        Args:
            allowed: Array of allowed token IDs.
            logits: Raw model output logits for all tokens.

        Returns:
            Masked logits array where disallowed tokens have -inf logits.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if allowed is None or allowed.size == 0:
            return logits
        masked = np.full_like(logits, -np.inf, dtype=np.float32)
        if allowed.size > 0:
            masked[allowed] = logits[allowed]
        return masked
