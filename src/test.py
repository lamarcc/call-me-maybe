import numpy as np

logits = np.array(
    [0.2, -0.5, 1.3, 0.0, -1.0, 2.1, -0.3, 0.7, -2.0, 0.5],
    dtype=np.float32
)

valid_tokens = np.array([1, 3, 5, 7], dtype=np.int32)

all_ids = np.arange(10, dtype=np.int32)

print(valid_tokens)
