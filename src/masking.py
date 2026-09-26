import numpy as np


class Mask():
    def __init__(self, agent) -> None:
        self.agent = agent

    def _get_token(self, text: str) -> np.ndarray:
        token = self.agent.encode(text).tolist()
        if isinstance(token[0], list):
            token = token[0]
        return np.array(token, dtype=np.int32)

    def get_int(self, value) -> np.ndarray:
        allowed = list("0123456789")
        if value == "":
            allowed.append("-")
        allowed_token = set()
        for c in allowed:
            allowed_token.update(self._get_token(c).tolist())
        return np.array(list(allowed_token), dtype=np.int32)

    def get_float(self, value) -> np.ndarray:
        allowed = list("0123456789")
        if "." not in value:
            allowed.append(".")
        if value == "":
            allowed.append("-")
        allowed_token = set()
        for c in allowed:
            allowed_token.update(self._get_token(c).tolist())
        return np.array(list(allowed_token), dtype=np.int32)

    def get_boolean(self) -> np.ndarray:
        allowed = set()
        for word in ["true", "false"]:
            allowed.update(self._get_token(word).tolist())
        return np.array(list(allowed), dtype=np.int32)

    def get_allowed_type(self, value_type, value) -> np.ndarray:
        if value_type == "integer":
            return self.get_int(value)
        if value_type == "float" or value_type == "number":
            return self.get_float(value)
        if value_type == "boolean":
            return self.get_boolean()
        if value_type == "string":
            return None
        raise ValueError(f"Unknown type: {value_type}")

    def mask_logits(self, allowed, logits) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        if allowed is None or allowed.size == 0:
            return logits
        masked = np.full_like(logits, -np.inf, dtype=np.float32)
        if allowed.size > 0:
            masked[allowed] = logits[allowed]
        return masked
