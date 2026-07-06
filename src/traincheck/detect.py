"""Best-effort stack detection for a config/script file.

Runs a fixed priority list of signatures and returns the first match:
HPC scheduler directives first (unambiguous and cheap to check), then
structured YAML shapes, then a generic shell fallback, then Python
launcher imports, then traincheck's own native schema.
"""

import re
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import yaml


class Stack(Enum):
    SLURM = "slurm"
    LSF = "lsf"
    PBS = "pbs"
    SGE = "sge"
    K8S_CRD = "k8s_crd"
    RAY = "ray"
    SKYPILOT = "skypilot"
    BARE = "bare"
    TORCHX = "torchx"
    SUBMITIT = "submitit"
    NATIVE = "native"
    UNKNOWN = "unknown"


# Priority order: HPC directives are checked first since they're the
# cheapest, least ambiguous signal - a shell script can otherwise look
# like anything.
_HPC_DIRECTIVES = (
    (re.compile(r"^\s*#SBATCH\b", re.MULTILINE), Stack.SLURM),
    (re.compile(r"^\s*#BSUB\b", re.MULTILINE), Stack.LSF),
    (re.compile(r"^\s*#PBS\b", re.MULTILINE), Stack.PBS),
    (re.compile(r"^\s*#\$", re.MULTILINE), Stack.SGE),
)

_K8S_CRD_KINDS = {"PyTorchJob", "MPIJob", "TFJob", "Job"}
_K8S_CRD_API_HINTS = ("kubeflow", "volcano")

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

_SHELL_LAUNCHER_RE = re.compile(r"\b(torchrun|accelerate)\b")
_TORCHX_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+torchx\b", re.MULTILINE)
_SUBMITIT_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+submitit\b", re.MULTILINE)


def detect_stack(path: Union[str, Path]) -> Stack:
    p = Path(path)

    if p.is_dir():
        candidates = sorted(f.name for f in p.iterdir() if f.is_file())
        raise IsADirectoryError(
            f"{p} is a directory, not a file - pick one of its entrypoints: {candidates}"
        )

    text = p.read_text()

    for pattern, stack in _HPC_DIRECTIVES:
        if pattern.search(text):
            return stack

    doc = _try_yaml(text)
    if isinstance(doc, dict):
        if _is_k8s_crd(doc):
            return Stack.K8S_CRD
        if _is_ray(doc):
            return Stack.RAY
        if _is_skypilot(doc):
            return Stack.SKYPILOT

    if _looks_like_shell(p, text) and _SHELL_LAUNCHER_RE.search(text):
        return Stack.BARE

    if _TORCHX_IMPORT_RE.search(text):
        return Stack.TORCHX
    if _SUBMITIT_IMPORT_RE.search(text):
        return Stack.SUBMITIT

    if isinstance(doc, dict) and _NATIVE_KEYS & doc.keys():
        return Stack.NATIVE

    return Stack.UNKNOWN


def _try_yaml(text: str) -> Optional[dict]:
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return doc if isinstance(doc, dict) else None


def _is_k8s_crd(doc: dict) -> bool:
    kind = doc.get("kind")
    api_version = str(doc.get("apiVersion", ""))
    return kind in _K8S_CRD_KINDS and any(hint in api_version for hint in _K8S_CRD_API_HINTS)


def _is_ray(doc: dict) -> bool:
    return doc.get("kind") == "RayJob" or "cluster_name" in doc


def _is_skypilot(doc: dict) -> bool:
    return "resources" in doc and "run" in doc


def _looks_like_shell(p: Path, text: str) -> bool:
    """Distinguish an actual shell script from other text that happens to
    mention torchrun/accelerate in passing (e.g. a torchx component's
    entrypoint="torchrun" string).
    """
    if p.suffix in (".sh", ".bash"):
        return True
    first_line = text.splitlines()[0] if text.splitlines() else ""
    return first_line.startswith("#!") and "sh" in first_line
