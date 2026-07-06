"""Best-effort stack detection for a config/script file.

Phase 1 (this module): just enough to route a file to the right adapter -
an sbatch script vs. traincheck's own native YAML schema. A fuller
detector covering the rest of the launcher ecosystem is Phase 2.
"""

import re
from enum import Enum
from pathlib import Path
from typing import Union

import yaml

_SBATCH_RE = re.compile(r"^\s*#SBATCH\b", re.MULTILINE)
_NATIVE_KEYS = {
    "cluster",
    "nccl",
    "framework",
    "parallelism",
    "environment",
    "model",
    "data",
    "checkpoint",
}


class Stack(Enum):
    SLURM = "slurm"
    NATIVE = "native"
    UNKNOWN = "unknown"


def detect_stack(path: Union[str, Path]) -> Stack:
    text = Path(path).read_text()

    if _SBATCH_RE.search(text):
        return Stack.SLURM

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        parsed = None

    if isinstance(parsed, dict) and _NATIVE_KEYS & parsed.keys():
        return Stack.NATIVE

    return Stack.UNKNOWN
