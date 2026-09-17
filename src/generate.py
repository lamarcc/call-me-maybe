from __future__ import annotations
from llm_sdk import Small_LLM_Model
from enum import Enum, auto
import numpy as np
import errors
import torch


agent = Small_LLM_Model()


class Vocab():
    path = agent.get_path_to_vocab_file()
