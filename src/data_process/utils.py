import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union

NUMBER = Union[int, float]

SIZE_OF_SENTENCE_EMBEDDING = 768
ZEROS_SENTENCE_EMBEDDING = np.zeros(SIZE_OF_SENTENCE_EMBEDDING)
PHONES_TIER = "phones"
PAD_TOKEN = "<PAD>"
NO_CONTEXT = "no_context"


def dump_json(filename: Path, data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent = 4)

def load_json(filename: Path):
    with open(filename) as infile:
        return json.load(infile)