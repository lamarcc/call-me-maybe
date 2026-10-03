from typing import Any
import numpy as np


class Mask():
    def __init__(self, agent: Any) -> None:
        self.agent = agent
        closers = ['"', '",', '"}', '"\n', '",\n', '}\n', '}', ',']
        tokens: set = set()
        for c in closers:
            tokens.update(self.get_token(c))
        self.closing_tokens: np.ndarray = np.array(
            list(tokens), dtype=np.int32
        )

    def get_token(self, text: str) -> np.ndarray:
        token = self.agent.encode(text).tolist()
        if isinstance(token[0], list):
            token = token[0]
        return np.array(token, dtype=np.int32)

    def _get_int(self, value: str) -> np.ndarray:
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
        allowed: set = set()
        for word in ["true", "false"]:
            allowed.update(self.get_token(word))
        allowed.update(self.closing_tokens.tolist())
        return np.array(list(allowed), dtype=np.int32)

    def _get_allowed_type(
        self, value_type: str, value: str
    ) -> np.ndarray | None:
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
        logits = np.asarray(logits, dtype=np.float32)
        if allowed is None or allowed.size == 0:
            return logits
        masked = np.full_like(logits, -np.inf, dtype=np.float32)
        if allowed.size > 0:
            masked[allowed] = logits[allowed]
        return masked
