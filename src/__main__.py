from llm_sdk import Small_LLM_Model
import models
from sys import argv


if __name__ == "__main__":
    path = ""
    if argv[1:]:
        path = argv[1]
    test = models.Parser(path)
    test.function()
